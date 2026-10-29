"""Kappa de Cohen por aspecto entre os anotadores A e B, conflitos e consolidação do ouro.

    python -m review_intel.labeling.kappa                                   # planilhas padrão em data/labels/
    python -m review_intel.labeling.kappa --a labeling/gold_anotador_A_preenchido.csv --b data/labels/gold_opus.csv \
        --agreement-only --name-a argon --name-b opus-5.5      # ouro = só os reviews em que A e B concordam
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from review_intel.core.io import LABELING_OUT, write_jsonl
from review_intel.core.metrics import kappa_per_aspect
from review_intel.core.schema import ASPECTS, LABELS


def read_annotations(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    for a in ASPECTS:
        df[a] = df[a].str.strip().str.lower()
    bad = df[~df[list(ASPECTS)].isin(LABELS).all(axis=1)]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} linhas com rótulo vazio ou inválido, ex.: {bad.review_id.iloc[0]}")
    return df.set_index("review_id")


def consolidate(a: pd.DataFrame, b: pd.DataFrame, adjudicated: pd.DataFrame | None = None):
    """Ouro = rótulo comum quando A e B concordam; nos conflitos, a coluna final_<aspecto> resolvida."""
    b = b.loc[a.index]
    gold, conflicts = [], []
    for rid in a.index:
        asp = {}
        for x in ASPECTS:
            if a.at[rid, x] == b.at[rid, x]:
                asp[x] = a.at[rid, x]
            elif adjudicated is not None and adjudicated.at[rid, f"final_{x}"] in LABELS:
                asp[x] = adjudicated.at[rid, f"final_{x}"]
            else:
                asp[x] = None
        if any(v is None for v in asp.values()):
            conflicts.append(rid)
        else:
            gold.append({"review_id": rid, "analysis": {"aspectos": asp,
                         "motivo_principal": a.get("motivo_principal", {}).get(rid, ""),
                         "acao_sugerida": a.get("acao_sugerida", {}).get(rid, "")}})
    return gold, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(LABELING_OUT / "gold_anotador_A.csv"))
    ap.add_argument("--b", default=str(LABELING_OUT / "gold_anotador_B.csv"))
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out-dir", default=str(LABELING_OUT), help="onde gravar gold.jsonl, kappa.json e descartados")
    ap.add_argument("--agreement-only", action="store_true",
                    help="sem adjudicação: o ouro fica só com os reviews em que A e B concordam nos 5 aspectos")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = read_annotations(args.a).fillna("")
    b = read_annotations(args.b)
    if set(a.index) != set(b.index):
        raise ValueError(f"os arquivos não cobrem os mesmos review_id ({len(a)} × {len(b)})")
    a = a.loc[sorted(a.index)]
    kappa = kappa_per_aspect(a[list(ASPECTS)].to_dict("records"), b.loc[a.index][list(ASPECTS)].to_dict("records"))
    agree = {x: float((a[x] == b.loc[a.index, x]).mean()) for x in ASPECTS}
    report = {"anotadores": [args.name_a, args.name_b], "n": len(a), "kappa": kappa, "concordancia_bruta": agree}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "kappa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    adj_path = out / "gold_conflitos.csv"
    adjudicated = None
    if not args.agreement_only and adj_path.exists():
        adjudicated = pd.read_csv(adj_path, dtype=str).fillna("").set_index("review_id")
    gold, conflicts = consolidate(a, b, adjudicated)

    rows = []
    for rid in conflicts:
        row = {"review_id": rid, "text": a.at[rid, "text"] if "text" in a else ""}
        for x in ASPECTS:
            row[f"{args.name_a}_{x}"], row[f"{args.name_b}_{x}"] = a.at[rid, x], b.at[rid, x]
            row[f"final_{x}"] = a.at[rid, x] if a.at[rid, x] == b.at[rid, x] else ""
        rows.append(row)

    if args.agreement_only:
        write_jsonl(out / "gold.jsonl", gold)
        pd.DataFrame(rows).to_csv(out / "gold_descartados.csv", index=False)
        print(f"ouro (só concordantes): {len(gold)} de {len(a)} reviews em {out / 'gold.jsonl'}; "
              f"{len(conflicts)} descartados em gold_descartados.csv")
    elif conflicts and adjudicated is None:
        pd.DataFrame(rows).to_csv(adj_path, index=False)
        print(f"{len(conflicts)} reviews com conflito em {adj_path}; preencha final_* e rode de novo")
    elif conflicts:
        print(f"ainda há {len(conflicts)} conflitos sem final_* preenchido")
    else:
        write_jsonl(out / "gold.jsonl", gold)
        print(f"ouro consolidado: {len(gold)} reviews em {out / 'gold.jsonl'}")


if __name__ == "__main__":
    main()
