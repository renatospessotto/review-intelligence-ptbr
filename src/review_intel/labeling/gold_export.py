"""Gera as planilhas de anotação do conjunto ouro, uma por anotador (ordens diferentes, sem a nota)."""
import pandas as pd

from review_intel.core.io import LABELING_OUT, PROCESSED
from review_intel.core.schema import ASPECTS


def main():
    df = pd.read_parquet(PROCESSED / "gold_candidates.parquet")[["review_id", "text"]]
    for col in [*ASPECTS, "motivo_principal", "acao_sugerida"]:
        df[col] = ""
    LABELING_OUT.mkdir(parents=True, exist_ok=True)
    for who, seed in [("A", 1), ("B", 2)]:
        path = LABELING_OUT / f"gold_anotador_{who}.csv"
        if path.exists():
            print(f"{path} já existe; não sobrescrevo anotação")
            continue
        df.sample(frac=1, random_state=seed).to_csv(path, index=False)
        print(f"gerado {path}")


if __name__ == "__main__":
    main()
