"""Empacota o que o Colab precisa (código + dados processados + rótulos) em dist/colab_bundle.zip.

Sobe esse zip para MyDrive/review-intelligence-ptbr/ e abra notebooks/03_treino_colab.ipynb no Colab.
"""
import zipfile
from pathlib import Path

from review_intel.core.io import ROOT

INCLUDE_DIRS = ["src", "data", "docs", "notebooks", "tests"]
INCLUDE_FILES = ["pyproject.toml", "README.md", "Makefile", "pytest.ini"]
SKIP_PARTS = {"__pycache__", "raw", ".pytest_cache"}
SKIP_NAMES = {".env", "smoke.jsonl"}  # a chave nunca vai no pacote; rótulos sintéticos não servem para treino


def files():
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and not (SKIP_PARTS & set(rel.parts)) and p.name not in SKIP_NAMES \
                    and not any(part.endswith(".egg-info") for part in rel.parts):
                yield p, rel
    for f in INCLUDE_FILES:
        yield ROOT / f, Path(f)


def main():
    out = ROOT / "dist" / "colab_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in files():
            z.write(p, rel)
    names = zipfile.ZipFile(out).namelist()
    assert "data/labels/silver.jsonl" in names and "data/processed/silver_pool.parquet" in names
    assert not any(n.endswith(".env") for n in names)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} arquivos)")


if __name__ == "__main__":
    main()
