"""Junta reviews, pedidos e clientes; limpa o texto e gera os splits disjuntos.

Saída em data/processed/: reviews.parquet (todos) e silver_pool / gold_candidates / holdout.parquet.
"""
import argparse
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from review_intel.core.io import PROCESSED, RAW

SEED = 42
MIN_CHARS = 5


def normalize(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", str(text))
    return re.sub(r"\s{2,}", " ", text).strip()


def build_reviews(raw_dir=RAW) -> pd.DataFrame:
    reviews = pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv")
    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")

    df = reviews.dropna(subset=["review_comment_message"]).drop_duplicates("review_id")
    title = df["review_comment_title"].fillna("").map(normalize)
    msg = df["review_comment_message"].map(normalize)
    df = df.assign(text=[f"{t}. {m}" if t else m for t, m in zip(title, msg)])
    df = df[df["text"].str.len() >= MIN_CHARS]
    # textos idênticos ("Ótimo", "Recomendo") repetem centenas de vezes e não ensinam nada novo
    df = df[~df["text"].str.lower().duplicated()]

    orders = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders["delay_days"] = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.days
    orders["delivered_late"] = orders["delay_days"] > 0
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)

    df = df.merge(
        orders[["order_id", "order_status", "customer_state", "delay_days", "delivered_late", "purchase_month"]],
        on="order_id",
        how="left",
    )
    keep = ["review_id", "order_id", "review_score", "text", "review_creation_date",
            "order_status", "customer_state", "delay_days", "delivered_late", "purchase_month"]
    return df[keep].reset_index(drop=True)


def make_splits(df: pd.DataFrame, n_silver: int, n_gold: int, seed: int = SEED):
    """Splits disjuntos e estratificados pela nota."""
    rest, gold = train_test_split(df, test_size=n_gold, stratify=df["review_score"], random_state=seed)
    holdout, silver = train_test_split(rest, test_size=n_silver, stratify=rest["review_score"], random_state=seed)
    return {
        "silver_pool": silver.reset_index(drop=True),
        "gold_candidates": gold.reset_index(drop=True),
        "holdout": holdout.reset_index(drop=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-silver", type=int, default=10_000)
    ap.add_argument("--n-gold", type=int, default=400)
    args = ap.parse_args()

    df = build_reviews()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "reviews.parquet", index=False)
    for name, part in make_splits(df, args.n_silver, args.n_gold).items():
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name}: {len(part)}")
    print(f"total: {len(df)} reviews com texto")


if __name__ == "__main__":
    main()
