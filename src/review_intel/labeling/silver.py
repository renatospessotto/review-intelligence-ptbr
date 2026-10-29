"""Rotulagem prata: Claude Haiku 5.5 rotula reviews com saída estruturada (destilação).

    python -m review_intel.labeling.silver --dry-run              # estima custo, não chama a API
    python -m review_intel.labeling.silver --limit 20             # teste rápido (síncrono)
    python -m review_intel.labeling.silver --mode batch           # Batch API (50% mais barato), tudo
    python -m review_intel.labeling.silver --mode batch --collect BATCH_ID

Resultados em data/labels/silver.jsonl (um por review_id). Reexecutar pula o que já foi rotulado.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from review_intel.core.io import LABELING_OUT, PROCESSED, append_jsonl, read_jsonl
from review_intel.core.claude_api import (BATCH_DISCOUNT, MODEL, PRICE_IN, PRICE_OUT, cost, record_from_message,  # noqa: F401
                           request_params)
from review_intel.core.prompts import SYSTEM

OUT = LABELING_OUT / "silver.jsonl"


def pending(df: pd.DataFrame, out=OUT) -> pd.DataFrame:
    done = {r["review_id"] for r in read_jsonl(out) if r.get("analysis")}
    return df[~df["review_id"].isin(done)]


def estimate(df: pd.DataFrame, batch: bool) -> dict:
    """Estimativa grosseira sem API: ~3,5 caracteres por token em PT-BR."""
    sys_tokens = len(SYSTEM) / 3.5
    in_tokens = sys_tokens * len(df) + df["text"].str.len().sum() / 3.5 + 20 * len(df)
    out_tokens = 250 * len(df)  # JSON (~80) + raciocínio curto
    usd = (in_tokens * PRICE_IN + out_tokens * PRICE_OUT) * (BATCH_DISCOUNT if batch else 1.0)
    return {"reviews": len(df), "input_tokens": int(in_tokens), "output_tokens": int(out_tokens), "usd": round(usd, 2)}


def label_sync(client, df: pd.DataFrame, workers: int = 8, out=OUT) -> list[dict]:
    import anthropic

    def one(row):
        try:
            msg = client.messages.create(**request_params(row.text))
            return record_from_message(row.review_id, msg, batch=False)
        except anthropic.BadRequestError as e:
            return {"review_id": row.review_id, "analysis": None, "error": e.message}

    records = []
    with ThreadPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(one, df.itertuples()), 1):
            append_jsonl(out, [rec])  # grava um a um: uma queda no meio não perde o que já foi pago
            records.append(rec)
            if i % 50 == 0:
                print(f"{i}/{len(df)}")
    return records


def submit_batch(client, df: pd.DataFrame) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [Request(custom_id=r.review_id, params=MessageCreateParamsNonStreaming(**request_params(r.text)))
                for r in df.itertuples()]
    batch = client.messages.batches.create(requests=requests)
    (LABELING_OUT / "batches.txt").parent.mkdir(parents=True, exist_ok=True)
    with (LABELING_OUT / "batches.txt").open("a") as f:
        f.write(batch.id + "\n")
    return batch.id


def collect_batch(client, batch_id: str, poll_s: int = 60, out=OUT) -> list[dict]:
    while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"{b.processing_status}: {b.request_counts.processing} em processamento")
        time.sleep(poll_s)
    records = []
    for res in client.messages.batches.results(batch_id):  # ordem arbitrária: chave é o custom_id
        if res.result.type == "succeeded":
            records.append(record_from_message(res.custom_id, res.result.message, batch=True))
        else:
            records.append({"review_id": res.custom_id, "analysis": None, "error": res.result.type})
    append_jsonl(out, records)
    return records


def summarize(records: list[dict]) -> str:
    ok = sum(1 for r in records if r.get("analysis"))
    usd = sum(r.get("cost_usd", 0) for r in records)
    return f"{ok}/{len(records)} válidos, US$ {usd:.4f} (US$ {1000 * usd / max(len(records), 1):.3f} por mil)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="silver_pool")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=["sync", "batch"], default="sync")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", metavar="BATCH_ID")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pending(pd.read_parquet(PROCESSED / f"{args.split}.parquet"))
    if args.limit:
        df = df.head(args.limit)
    if args.dry_run:
        print(json.dumps(estimate(df, batch=args.mode == "batch"), indent=2))
        return

    import anthropic

    from review_intel.core.env import load_dotenv

    load_dotenv()
    client = anthropic.Anthropic()
    t0 = time.time()
    if args.collect:
        records = collect_batch(client, args.collect)
    elif args.mode == "batch":
        batch_id = submit_batch(client, df)
        print(f"batch {batch_id} enviado; colete com --collect {batch_id}")
        return
    else:
        records = label_sync(client, df, workers=args.workers)
    print(summarize(records), f"em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
