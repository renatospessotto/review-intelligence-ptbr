"""Rótulos SINTÉTICOS por regras, só para testar o pipeline de ponta a ponta sem a API.

Não use estes rótulos para nenhum resultado: são palavras-chave + nota, com muito erro.
Saída: data/labels/smoke.jsonl, no mesmo formato de silver.jsonl.
"""
import argparse
import re

import pandas as pd

from review_intel.core.io import LABELING_OUT, PROCESSED, write_jsonl
from review_intel.core.schema import ASPECTS

KEYWORDS = {
    "entrega": r"entreg|cheg|prazo|atras|demor|rápid|correio|receb",
    "produto": r"produto|qualidade|defeit|quebr|funcion|ótimo|excelente|bom|ruim|péssim|adorei|recomendo",
    "atendimento": r"atendiment|vendedor|loja|contato|respost|suporte|sac",
    "preco": r"preço|barat|caro|custo|frete|valor",
    "embalagem": r"embala|caixa|amass|lacrad|protegid",
}
NEG = r"\bnão\b|nunca|atras|demor|defeit|quebr|ruim|péssim|horrível|errad|faltou|amass|caro"


def label(text: str, score: int) -> dict:
    t = text.lower()
    polarity = "negativo" if (score <= 2 or re.search(NEG, t)) else ("neutro" if score == 3 else "positivo")
    asp = {a: (polarity if re.search(kw, t) else "ausente") for a, kw in KEYWORDS.items()}
    if all(v == "ausente" for v in asp.values()):
        asp["produto"] = polarity
    return {"aspectos": asp, "motivo_principal": text[:60], "acao_sugerida": "nenhuma" if score >= 4 else "revisar pedido"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["silver_pool", "gold_candidates"])
    args = ap.parse_args()
    rows = []
    for split in args.splits:
        df = pd.read_parquet(PROCESSED / f"{split}.parquet")
        rows += [{"review_id": r.review_id, "analysis": label(r.text, r.review_score), "model": "SMOKE-regras"}
                 for r in df.itertuples()]
    write_jsonl(LABELING_OUT / "smoke.jsonl", rows)
    print(f"{len(rows)} rótulos sintéticos em {LABELING_OUT / 'smoke.jsonl'}")
    assert all(set(r["analysis"]["aspectos"]) == set(ASPECTS) for r in rows)


if __name__ == "__main__":
    main()
