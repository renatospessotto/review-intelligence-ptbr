"""Cascata calibrada: o Laya responde; se a confiança de algum aspecto ficar abaixo do limiar, o review vai para o LLM.

    python -m review_intel.evaluation.cascade --cheap laya-ft --fallback claude

Lê as predições salvas por review_intel.evaluation.run e reporta ECE do modelo barato, a curva F1 × custo por limiar
e a fração de chamadas ao LLM economizadas.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from review_intel.evaluation.run import EVAL_OUT
from review_intel.core.data import load_labeled
from review_intel.core.io import GOLD, read_jsonl
from review_intel.core.metrics import aspect_f1, ece
from review_intel.core.schema import ASPECTS, aspect_labels


def route(cheap: list[dict], fallback: list[dict], threshold: float) -> tuple[list[dict], np.ndarray]:
    """Para cada review, usa o modelo barato se min(confiança) >= limiar; senão, o fallback."""
    escalate = np.array([min(c["confidence"].values()) < threshold if c["confidence"] else True for c in cheap])
    chosen = [f if e else c for c, f, e in zip(cheap, fallback, escalate)]
    return chosen, escalate


def calibration(cheap: list[dict], gold: list[dict]) -> float:
    conf, correct = [], []
    for c, g in zip(cheap, gold):
        labels = aspect_labels(c["analysis"])
        for a in ASPECTS:
            conf.append(c["confidence"][a])
            correct.append(labels[a] == g[a])
    return ece(conf, correct)


def sweep(cheap, fallback, gold, thresholds) -> pd.DataFrame:
    cheap_cost = np.array([c["cost_usd"] for c in cheap])
    fb_cost = np.array([f["cost_usd"] for f in fallback])
    rows = []
    for t in thresholds:
        chosen, esc = route(cheap, fallback, t)
        f1 = aspect_f1(gold, [aspect_labels(x["analysis"]) for x in chosen])["macro"]
        cost = cheap_cost.sum() + fb_cost[esc].sum()  # o barato sempre roda; o LLM só quando escala
        rows.append({"limiar": t, "F1 macro": f1, "% escalado": esc.mean(), "% chamadas ao LLM economizadas": 1 - esc.mean(),
                     "US$ / mil": 1000 * cost / len(cheap)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", default="laya-ft")
    ap.add_argument("--fallback", default="claude")
    ap.add_argument("--split", default="gold_candidates")
    ap.add_argument("--labels", default=str(GOLD))
    ap.add_argument("--pred-dir", default=str(EVAL_OUT / "predictions"))
    args = ap.parse_args()

    df = load_labeled(args.split, args.labels)
    gold_by_id = {r.review_id: {a: getattr(r, a) for a in ASPECTS} for r in df.itertuples()}
    cheap = {r["review_id"]: r for r in read_jsonl(Path(args.pred_dir) / f"{args.cheap}.jsonl")}
    fb = {r["review_id"]: r for r in read_jsonl(Path(args.pred_dir) / f"{args.fallback}.jsonl")}
    ids = [i for i in cheap if i in fb and i in gold_by_id]
    if not ids:
        raise SystemExit("sem predições em comum; rode review_intel.evaluation.run com as duas abordagens antes")
    cheap_l, fb_l, gold = [cheap[i] for i in ids], [fb[i] for i in ids], [gold_by_id[i] for i in ids]

    print(f"ECE do {args.cheap}: {calibration(cheap_l, gold):.3f}  ({len(ids)} reviews)")
    table = sweep(cheap_l, fb_l, gold, np.round(np.arange(0.0, 1.0001, 0.05), 2))
    out = Path(args.pred_dir).parent
    table.to_csv(out / f"cascade_{args.cheap}_{args.fallback}.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
