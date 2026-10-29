"""Baixa o Olist Brazilian E-Commerce pela API do Kaggle (precisa de ~/.kaggle/kaggle.json)."""
from review_intel.core.io import RAW

DATASET = "olistbr/brazilian-ecommerce"


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / "olist_order_reviews_dataset.csv").exists():
        print(f"já baixado em {RAW}")
        return
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(RAW), unzip=True)
    print(f"baixado em {RAW}")


if __name__ == "__main__":
    main()
