"""SFT com LoRA de um LLM pequeno nos rótulos prata.

    # Colab (T4/L4): QLoRA 4 bits
    python -m review_intel.training.qlora --backend cuda --model Qwen/Qwen3.5-2B --labels data/labels/silver.jsonl
    # Mac: LoRA bf16 sem quantização, só smoke
    python -m review_intel.training.qlora --backend mps --model Qwen/Qwen3.5-0.8B --labels data/labels/smoke.jsonl --limit 50 --max-steps 10
"""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from review_intel.training.sft_data import to_sft_dataset
from review_intel.core.data import load_labeled, split_train_val
from review_intel.core.io import OUTPUTS


def load_model(model_id: str, backend: str):
    if backend == "cuda":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
        return prepare_model_for_kbit_training(model)
    dtype = torch.float32 if backend == "cpu" else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--labels", default="data/labels/silver.jsonl")
    ap.add_argument("--split", default="silver_pool")
    ap.add_argument("--backend", choices=["cuda", "mps", "cpu"], default="cuda")
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.output or OUTPUTS / f"lora-{args.model.split('/')[-1]}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df = load_labeled(args.split, args.labels)
    if args.limit:
        df = df.head(args.limit)
    train, val = split_train_val(df)
    print(f"treino {len(train)} | validação {len(val)}")

    model = load_model(args.model, args.backend)
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        logging_steps=5,
        eval_strategy="steps" if args.max_steps < 0 else "no",
        eval_steps=100,
        save_strategy="no",
        max_length=512,
        completion_only_loss=True,  # perda só no JSON, não no prompt
        bf16=args.backend == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=args.backend == "cuda" and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.backend == "cuda",
        report_to="none",
        seed=42,
    )
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=to_sft_dataset(train, tok),
                         eval_dataset=to_sft_dataset(val, tok), processing_class=tok, peft_config=lora)
    result = trainer.train()
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    summary = {"base_model": args.model, "labels": args.labels, "n_train": len(train),
               "train_loss": result.training_loss, "steps": result.global_step, "backend": args.backend}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
