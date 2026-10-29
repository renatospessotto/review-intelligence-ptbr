"""Fine-tuning do Laya multilingual nos rótulos prata, com o laço de treino do próprio pacote (laya.train).

    python -m review_intel.training.laya --labels data/labels/silver.jsonl
    python -m review_intel.training.laya --labels data/labels/smoke.jsonl --limit 64 --epochs 1   # smoke

O checkpoint salvo carrega com LayaPredictor(checkpoint=<pasta>, subfolder=None).
"""
import argparse
import json
from pathlib import Path

from review_intel.core.data import load_labeled, split_train_val
from review_intel.core.io import OUTPUTS, write_jsonl
from review_intel.models.laya import questions
from review_intel.core.schema import ASPECTS


def to_rows(df) -> list[dict]:
    """Formato {state, questions, gold} do laya.train: gold é uma distribuição one-hot por aspecto."""
    q = questions()
    return [{"state": r.text, "questions": q,
             "gold": {a: {"probabilities": {getattr(r, a): 1.0}} for a in ASPECTS}}
            for r in df.itertuples()]


def base_checkpoint_dir() -> str:
    from huggingface_hub import snapshot_download

    root = snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"])
    return str(Path(root) / "multilingual")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/silver.jsonl")
    ap.add_argument("--split", default="silver_pool")
    ap.add_argument("--output", default=str(OUTPUTS / "laya-ft"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from laya.train import TrainConfig, finetune

    df = load_labeled(args.split, args.labels)
    if args.limit:
        df = df.head(args.limit)
    train, val = split_train_val(df)
    data_dir = Path(args.output) / "data"
    write_jsonl(data_dir / "train.jsonl", to_rows(train))
    write_jsonl(data_dir / "val.jsonl", to_rows(val))

    cfg = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                      loss="soft-ce", shuffle_options=("choice",), eval_data=str(data_dir / "val.jsonl"),
                      log_every=10)
    summary = finetune(str(data_dir / "train.jsonl"), base_checkpoint_dir(), args.output, cfg, device=args.device)
    (Path(args.output) / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=2, default=str))


if __name__ == "__main__":
    main()
