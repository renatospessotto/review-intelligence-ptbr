"""Descompacta o zip de modelos treinados no Colab em outputs/ (lora-*, laya-ft, bertimbau-multiaspecto).

    python -m review_intel.hub.import_models ~/Downloads/modelos_treinados.zip
"""
import sys
import zipfile

from review_intel.core.io import OUTPUTS


def main():
    path = sys.argv[1]
    with zipfile.ZipFile(path) as z:
        tops = sorted({n.split("/")[0] for n in z.namelist()})
        OUTPUTS.mkdir(exist_ok=True)
        z.extractall(OUTPUTS)
    print(f"extraído em {OUTPUTS}: {', '.join(tops)}")


if __name__ == "__main__":
    main()
