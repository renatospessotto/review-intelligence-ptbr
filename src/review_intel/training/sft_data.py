"""Rótulos (prata ou smoke) → dataset prompt/completion em texto, com o mesmo template da inferência."""
import json

from datasets import Dataset

from review_intel.core.prompts import chat_messages


def build_prompt(tok, text: str) -> str:
    return tok.apply_chat_template(chat_messages(text), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def to_sft_dataset(df, tok) -> Dataset:
    """Uma linha por review: `prompt` (system+user formatados) e `completion` (JSON + fim de turno)."""
    return Dataset.from_dict({
        "prompt": [build_prompt(tok, t) for t in df["text"]],
        "completion": [json.dumps(a, ensure_ascii=False) + tok.eos_token for a in df["analysis"]],
    })
