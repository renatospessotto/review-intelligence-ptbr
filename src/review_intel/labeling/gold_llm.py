"""Anotador cego do conjunto ouro via API: reproduz o anotador Opus 5.5 sem o Claude Code.

O modelo recebe o guia de anotação inteiro como system prompt e um review por vez. Não vê rótulos
de outros anotadores nem a nota do review. Saída no mesmo formato de data/gold/annotations/*.csv.

    python -m review_intel.labeling.gold_llm --dry-run                     # estimativa de custo, sem chamar a API
    python -m review_intel.labeling.gold_llm --limit 5                     # teste rápido
    python -m review_intel.labeling.gold_llm --model claude-opus-5-5 --out data/gold/annotations/opus-5.5-api.csv

Sem fallback de modelo: se o modelo recusar, o review fica sem rótulo, em vez de ser anotado por
outro modelo, o que misturaria anotadores no mesmo arquivo.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from review_intel.core.io import PROCESSED, ROOT
from review_intel.core.schema import ASPECTS, LABELS

GUIDE = (ROOT / "docs" / "guia_anotacao.md").read_text(encoding="utf-8")
SYSTEM = f"""Você é um anotador do conjunto ouro descrito no guia abaixo. Anote cada avaliação com cuidado, \
seguindo as definições, a tabela "Inclui / Não inclui" e os casos de borda. Ignore as seções do guia sobre \
planilhas e fluxo: aqui você recebe uma avaliação por mensagem e responde só com os 5 rótulos.

<guia>
{GUIDE}
</guia>"""

# US$ por milhão de tokens (entrada, saída), tabela de preços de out/2026
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-fable-5-1": (10.0, 50.0)}


def schema() -> dict:
    return {
        "type": "object",
        "properties": {a: {"type": "string", "enum": list(LABELS)} for a in ASPECTS},
        "required": list(ASPECTS),
        "additionalProperties": False,
    }


def request_params(model: str, text: str, effort: str) -> dict:
    return {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Avaliação:\n{text.strip()}"}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema()}},
    }


def annotate_one(client, model: str, review_id: str, text: str, effort: str) -> dict:
    msg = client.messages.create(**request_params(model, text, effort))
    row = {"review_id": review_id, "_in": msg.usage.input_tokens, "_out": msg.usage.output_tokens}
    if msg.stop_reason != "end_turn":
        return {**row, **{a: "" for a in ASPECTS}, "_stop": msg.stop_reason}
    labels = json.loads(next(b.text for b in msg.content if b.type == "text"))
    return {**row, **{a: labels[a] for a in ASPECTS}, "_stop": msg.stop_reason}


def estimate(df: pd.DataFrame, model: str) -> dict:
    """Sem API: ~3,5 caracteres por token; saída ~300 tokens (JSON + raciocínio adaptativo)."""
    pin, pout = PRICES[model]
    tin = (len(SYSTEM) / 3.5 + 30) * len(df) + df["text"].str.len().sum() / 3.5
    tout = 300 * len(df)
    return {"model": model, "reviews": len(df), "usd": round((tin * pin + tout * pout) / 1e6, 2)}


def run(client, df: pd.DataFrame, model: str, effort: str, workers: int) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda r: annotate_one(client, model, r.review_id, r.text, effort), df.itertuples()))
    out = df[["review_id", "text"]].merge(pd.DataFrame(rows), on="review_id")
    pin, pout = PRICES.get(model, (0, 0))
    usd = (out["_in"].sum() * pin + out["_out"].sum() * pout) / 1e6
    print(f"{(out['_stop'] == 'end_turn').sum()}/{len(out)} anotados, US$ {usd:.3f}")
    return out.drop(columns=["_in", "_out", "_stop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--out", default=str(ROOT / "data" / "gold" / "annotations" / "opus-5.5-api.csv"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "gold_candidates.parquet")[["review_id", "text"]]
    if args.limit:
        df = df.head(args.limit)
    if args.dry_run:
        print(json.dumps(estimate(df, args.model), indent=2))
        return

    import anthropic

    from review_intel.core.env import load_dotenv

    load_dotenv()
    out = run(anthropic.Anthropic(), df, args.model, args.effort, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"salvo em {args.out}")


if __name__ == "__main__":
    main()
