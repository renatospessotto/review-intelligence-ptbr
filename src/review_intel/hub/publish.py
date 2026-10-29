"""Publica os modelos ajustados e o dataset no Hugging Face (repositórios públicos).

    python -m review_intel.hub.publish --dry-run              # lista arquivos e tamanhos, não envia nada
    python -m review_intel.hub.publish                        # tudo
    python -m review_intel.hub.publish --only laya dataset

Token: HF_TOKEN no .env, com permissão de escrita. Os model cards vêm de docs/model_cards/.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from review_intel.core.data import load_labeled, split_train_val
from review_intel.core.env import load_dotenv
from review_intel.core.io import GOLD, LABELING_OUT, OUTPUTS, PROCESSED, ROOT, read_jsonl, write_jsonl
from review_intel.core.prompts import SYSTEM
from review_intel.core.schema import ASPECTS
from review_intel.models.registry import HUB, HUB_DATASET

CARDS = ROOT / "docs" / "model_cards"
STAGING = OUTPUTS / "hub_staging"
TARGETS = ["laya", "bertimbau", "qwen", "dataset"]


def stage_laya() -> Path:
    src, dst = OUTPUTS / "laya-ft", STAGING / "laya"
    shutil.rmtree(dst, ignore_errors=True)
    # data/ (JSONL de treino) fica de fora: o dataset tem repositório próprio
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("data", "checkpoint_*"))
    return dst


def stage_bertimbau() -> Path:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    src, dst = OUTPUTS / "bertimbau-multiaspecto", STAGING / "bertimbau"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    state = torch.load(src / "model.pt", map_location="cpu")
    save_file({k: v.contiguous() for k, v in state.items()}, dst / "model.safetensors")
    cfg = json.loads((src / "config.json").read_text())
    summary = json.loads((src / "train_summary.json").read_text())
    cfg["temperatures"] = summary.get("temperatures")
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    shutil.copy(src / "train_summary.json", dst)
    AutoTokenizer.from_pretrained(cfg["base"]).save_pretrained(dst)
    return dst


def stage_qwen() -> Path:
    src, dst = OUTPUTS / "lora-Qwen3.5-2B", STAGING / "qwen"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*", "README.md"))
    (dst / "system_prompt.txt").write_text(SYSTEM, encoding="utf-8")
    return dst


def _flat(df: pd.DataFrame, extra: bool) -> list[dict]:
    rows = []
    for r in df.itertuples():
        row = {"review_id": r.review_id, "text": r.text, "review_score": int(r.review_score),
               **{a: getattr(r, a) for a in ASPECTS}}
        if extra:
            row |= {"motivo_principal": r.analysis.get("motivo_principal", ""),
                    "acao_sugerida": r.analysis.get("acao_sugerida", "")}
        rows.append(row)
    return rows


def stage_dataset() -> Path:
    dst = STAGING / "dataset"
    shutil.rmtree(dst, ignore_errors=True)
    write_jsonl(dst / "gold" / "test.jsonl", _flat(load_labeled("gold_candidates", GOLD), extra=False))
    train, val = split_train_val(load_labeled("silver_pool", LABELING_OUT / "silver.jsonl"))  # mesmo corte do treino
    write_jsonl(dst / "silver" / "train.jsonl", _flat(train, extra=True))
    write_jsonl(dst / "silver" / "validation.jsonl", _flat(val, extra=True))
    gold_dir = ROOT / "data" / "gold"
    shutil.copytree(gold_dir / "annotations", dst / "gold" / "annotations")
    shutil.copy(gold_dir / "kappa.json", dst / "gold")
    shutil.copy(gold_dir / "gold_descartados.csv", dst / "gold" / "descartados.csv")
    return dst


STAGE = {"laya": stage_laya, "bertimbau": stage_bertimbau, "qwen": stage_qwen, "dataset": stage_dataset}
REPO = {"laya": (HUB["laya-ft"], "model"), "bertimbau": (HUB["bertimbau"], "model"),
        "qwen": (HUB["small-lora"], "model"), "dataset": (HUB_DATASET, "dataset")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=TARGETS, default=TARGETS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for name in args.only:
        folder = STAGE[name]()
        shutil.copy(CARDS / f"{name}.md", folder / "README.md")
        repo_id, kind = REPO[name]
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"\n{kind} {repo_id}: {len(files)} arquivos, {size:.0f} MB")
        for p in files:
            print(f"  {p.relative_to(folder)}  ({p.stat().st_size / 1e6:.1f} MB)")
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type=kind, private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=folder,
                          commit_message="Publica modelo/dataset do projeto Review Intelligence PT-BR")
        prefix = "datasets/" if kind == "dataset" else ""
        print(f"  publicado: https://huggingface.co/{prefix}{repo_id}")


if __name__ == "__main__":
    main()
