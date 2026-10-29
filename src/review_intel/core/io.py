"""Caminhos e leitura/escrita de JSONL."""
import json
import os
from pathlib import Path

# raiz do repositório (src/review_intel/core/io.py -> 3 níveis acima); no Docker vem de RI_ROOT
ROOT = Path(os.environ.get("RI_ROOT") or Path(__file__).resolve().parents[3])
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LABELING_OUT = ROOT / "data" / "labels"
OUTPUTS = ROOT / "outputs"
GOLD = ROOT / "data" / "gold" / "gold.jsonl"  # ouro de consenso, versionado


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
