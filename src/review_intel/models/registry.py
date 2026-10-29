"""Abordagens disponíveis pelo nome usado no eval, na API e no notebook."""
import os
from pathlib import Path

ALL = ["small-zs", "small-lora", "laya-zs", "laya-ft", "bertimbau", "claude"]

# modelos publicados no Hugging Face (usados quando não há pasta local em outputs/)
HF_USER = "RenatoSpessotto"
HUB = {
    "laya-ft": f"{HF_USER}/laya-aspectos-reviews-ptbr",
    "bertimbau": f"{HF_USER}/bertimbau-aspectos-reviews-ptbr",
    "small-lora": f"{HF_USER}/qwen3.5-2b-aspectos-reviews-ptbr-lora",
}
HUB_DATASET = f"{HF_USER}/olist-aspectos-reviews-ptbr"


LOCAL = {"laya-ft": "laya-ft", "bertimbau": "bertimbau-multiaspecto", "small-lora": "lora-Qwen3.5-2B"}


def default_source(name: str) -> str:
    """Pasta em outputs/ se o modelo foi treinado aqui; senão o repositório publicado no Hub."""
    from review_intel.core.io import OUTPUTS

    local = OUTPUTS / LOCAL[name]
    return str(local) if local.exists() else HUB[name]


def resolve(path_or_repo: str) -> Path | None:
    """Pasta local, se existir; senão baixa do Hub quando parece um id `usuario/repo`."""
    if Path(path_or_repo).exists():
        return Path(path_or_repo)
    if path_or_repo.count("/") == 1 and not path_or_repo.startswith((".", "/", "outputs")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(path_or_repo))
    return None


def build(name: str, args):
    """Instancia a abordagem, ou devolve o motivo de não poder rodá-la."""
    if name == "small-zs":
        from review_intel.models.qwen import HFGenerator
        return HFGenerator(args.small_model)
    if name == "small-lora":
        if not (path := resolve(args.lora)):
            return f"adapter não encontrado em {args.lora}"
        from review_intel.models.qwen import HFGenerator
        return HFGenerator(args.small_model, adapter=str(path))
    if name == "laya-zs":
        from review_intel.models.laya import LayaPredictor
        return LayaPredictor()
    if name == "laya-ft":
        if not (path := resolve(args.laya_ft)):
            return f"checkpoint não encontrado em {args.laya_ft}"
        from review_intel.models.laya import LayaPredictor
        return LayaPredictor(str(path), subfolder=None)
    if name == "bertimbau":
        if not (path := resolve(args.bertimbau)):
            return f"modelo não encontrado em {args.bertimbau}"
        from review_intel.models.bertimbau import BertimbauPredictor
        return BertimbauPredictor(path)
    if name == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return "sem credencial da Anthropic no ambiente"
        from review_intel.models.claude import ClaudePredictor
        return ClaudePredictor()
    raise ValueError(name)


def load(name: str):
    """Instancia uma abordagem com as fontes padrão (variáveis RI_* > outputs/ > Hugging Face). Erro se indisponível."""
    from argparse import Namespace

    args = Namespace(
        small_model=os.environ.get("RI_SMALL_MODEL", "Qwen/Qwen3.5-2B"),
        lora=os.environ.get("RI_LORA") or default_source("small-lora"),
        laya_ft=os.environ.get("RI_LAYA_FT") or default_source("laya-ft"),
        bertimbau=os.environ.get("RI_BERTIMBAU") or default_source("bertimbau"),
    )
    predictor = build(name, args)
    if isinstance(predictor, str):
        raise RuntimeError(predictor)
    return predictor
