"""Esquema de saída: um rótulo por aspecto, mais motivo e ação em texto curto."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

ASPECTS = ("entrega", "produto", "atendimento", "preco", "embalagem")
LABELS = ("positivo", "negativo", "neutro", "ausente")

Label = Literal["positivo", "negativo", "neutro", "ausente"]


class Aspectos(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entrega: Label
    produto: Label
    atendimento: Label
    preco: Label
    embalagem: Label


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspectos: Aspectos
    motivo_principal: str
    acao_sugerida: str


def json_schema() -> dict:
    """Schema no formato aceito por `output_config.format` (sem $defs, additionalProperties=False)."""
    aspect_props = {a: {"type": "string", "enum": list(LABELS)} for a in ASPECTS}
    return {
        "type": "object",
        "properties": {
            "aspectos": {
                "type": "object",
                "properties": aspect_props,
                "required": list(ASPECTS),
                "additionalProperties": False,
            },
            "motivo_principal": {"type": "string"},
            "acao_sugerida": {"type": "string"},
        },
        "required": ["aspectos", "motivo_principal", "acao_sugerida"],
        "additionalProperties": False,
    }


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_output(text: str) -> ReviewAnalysis | None:
    """Extrai e valida o JSON gerado por um modelo. Devolve None se for inválido.

    Aceita bloco ```json```, texto antes/depois do objeto e bloco <think> de modelos de raciocínio.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return ReviewAnalysis.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None


def aspect_labels(analysis: ReviewAnalysis | dict | None) -> dict[str, str | None]:
    """Rótulos por aspecto; None em todos quando a saída é inválida."""
    if analysis is None:
        return {a: None for a in ASPECTS}
    if isinstance(analysis, ReviewAnalysis):
        analysis = analysis.model_dump()
    return {a: analysis["aspectos"][a] for a in ASPECTS}
