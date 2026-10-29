"""Junta textos de um split com rótulos (prata, ouro ou smoke) num DataFrame."""
from pathlib import Path

import pandas as pd

from review_intel.core.io import PROCESSED, read_jsonl
from review_intel.core.schema import ASPECTS


def load_labeled(split: str, labels: str | Path) -> pd.DataFrame:
    """Linhas do split que têm rótulo válido; uma coluna por aspecto e `analysis` com o JSON completo."""
    df = pd.read_parquet(PROCESSED / f"{split}.parquet")
    lab = {r["review_id"]: r["analysis"] for r in read_jsonl(labels) if r.get("analysis")}
    df = df[df["review_id"].isin(lab)].copy()
    df["analysis"] = df["review_id"].map(lab)
    for a in ASPECTS:
        df[a] = df["analysis"].map(lambda x, a=a: x["aspectos"][a])
    return df.reset_index(drop=True)


def split_train_val(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    val = df.sample(frac=val_frac, random_state=seed)
    return df.drop(val.index).reset_index(drop=True), val.reset_index(drop=True)
