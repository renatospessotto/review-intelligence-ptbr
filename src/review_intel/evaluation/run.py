"""Gera a tabela comparativa: F1 por aspecto, JSON válido, latência e custo por mil reviews.

    python -m review_intel.evaluation.run                                     # conjunto ouro, todas as abordagens disponíveis
    python -m review_intel.evaluation.run --labels data/labels/smoke.jsonl --limit 40 --approaches laya-zs small-zs
    python -m review_intel.evaluation.run --silver-val --approaches small-zs small-lora laya-zs laya-ft bertimbau   # 1.000 reviews de validação prata

Abordagens sem artefato (adapter, checkpoint, chave da API) são puladas com aviso.
Saídas em outputs/eval/: results.md, results.csv e predictions/<abordagem>.jsonl (usadas pela cascata).
"""
import argparse
from pathlib import Path

import pandas as pd

from review_intel.core.data import load_labeled, split_train_val
from review_intel.core.env import load_dotenv
from review_intel.core.io import GOLD, LABELING_OUT, OUTPUTS, read_jsonl, write_jsonl
from review_intel.core.metrics import aspect_f1, latency_stats
from review_intel.models.registry import ALL, build
from review_intel.core.predictors import Prediction
from review_intel.core.schema import ASPECTS, aspect_labels

EVAL_OUT = OUTPUTS / "eval"


def score(name: str, df: pd.DataFrame, preds: list[Prediction], secs: list[float], gpu_usd_hour: float) -> dict:
    gold = [{a: r[a] for a in ASPECTS} for _, r in df.iterrows()]
    pred_labels = [aspect_labels(p.analysis) for p in preds]
    f1 = aspect_f1(gold, pred_labels)
    api_cost = sum(p.cost_usd for p in preds)
    # modelos locais: custo = tempo de GPU alugada; API: custo real dos tokens
    cost = api_cost if api_cost > 0 else sum(secs) * gpu_usd_hour / 3600
    return {
        "abordagem": name,
        **{f"F1 {a}": f1[a] for a in ASPECTS},
        "F1 macro": f1["macro"],
        "JSON válido": sum(p.analysis is not None for p in preds) / len(preds),
        **latency_stats(secs),
        "US$ / mil": 1000 * cost / len(preds),
    }


def evaluate(predictor, df: pd.DataFrame, gpu_usd_hour: float, batch_size: int) -> tuple[dict, list[dict]]:
    preds, secs = predictor.timed_predict(list(df["text"]), batch_size=batch_size)
    row = score(predictor.name, df, preds, secs, gpu_usd_hour)
    api_cost = sum(p.cost_usd for p in preds)
    records = [{"review_id": rid, "name": predictor.name, "analysis": p.analysis, "confidence": p.confidence,
                "cost_usd": p.cost_usd if api_cost > 0 else s * gpu_usd_hour / 3600, "seconds": s}
               for rid, p, s in zip(df["review_id"], preds, secs)]
    return row, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gold_candidates")
    ap.add_argument("--labels", default=str(GOLD))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--approaches", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--small-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--lora", default=str(OUTPUTS / "lora-Qwen3.5-2B"))
    ap.add_argument("--laya-ft", default=str(OUTPUTS / "laya-ft"))
    ap.add_argument("--bertimbau", default=str(OUTPUTS / "bertimbau-multiaspecto"))
    ap.add_argument("--gpu-usd-hour", type=float, default=0.35, help="preço de referência de uma T4/L4 alugada")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="padrão: outputs/eval (ouro) ou outputs/eval/silver_val")
    ap.add_argument("--silver-val", action="store_true",
                    help="avalia nos 1.000 reviews de validação prata (rótulos do Haiku), o mesmo corte usado no treino")
    ap.add_argument("--from-predictions", action="store_true",
                    help="não roda os modelos: reaproveita out/predictions/<abordagem>.jsonl e só recalcula as métricas")
    args = ap.parse_args()
    load_dotenv()

    if args.silver_val:
        args.split, args.labels = "silver_pool", str(LABELING_OUT / "silver.jsonl")
        # mesmo corte de finetune/*.py: load_labeled + split_train_val(seed=42)
        _, df = split_train_val(load_labeled(args.split, args.labels))
        args.out = args.out or str(EVAL_OUT / "silver_val")
    else:
        df = load_labeled(args.split, args.labels)
        args.out = args.out or str(EVAL_OUT)
    if df.empty:
        raise SystemExit(f"nenhum review rotulado em {args.labels} para o split {args.split}")
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out)
    scope = "validação prata" if args.silver_val else args.split
    print(f"avaliando em {len(df)} reviews ({scope}, rótulos {args.labels})")

    rows = []
    for name in args.approaches:
        if args.from_predictions:
            saved = {r["review_id"]: r for r in read_jsonl(out / "predictions" / f"{name}.jsonl")}
            if not set(df["review_id"]) <= set(saved):
                print(f"[pulado] {name}: sem predições salvas para todos os reviews")
                continue
            recs = [saved[i] for i in df["review_id"]]
            preds = [Prediction(r["analysis"], r["confidence"], r["cost_usd"]) for r in recs]
            label = recs[0].get("name", name)
            rows.append({"id": name, **score(label, df, preds, [r["seconds"] for r in recs], args.gpu_usd_hour)})
            continue
        predictor = build(name, args)
        if isinstance(predictor, str):
            print(f"[pulado] {name}: {predictor}")
            continue
        print(f"[rodando] {predictor.name}")
        row, records = evaluate(predictor, df, args.gpu_usd_hour, args.batch_size)
        write_jsonl(out / "predictions" / f"{name}.jsonl", records)
        rows.append({"id": name, **row})
        del predictor

    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)
    header = f"Avaliação em {len(df)} reviews ({scope}, rótulos `{Path(args.labels).name}`)\n\n"
    (out / "results.md").write_text(header + table.drop(columns="id").to_markdown(index=False, floatfmt=".3f"))
    print(table.drop(columns="id").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
