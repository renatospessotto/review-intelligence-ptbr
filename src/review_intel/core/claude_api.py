"""Request do Claude usado na rotulagem prata e no ClaudePredictor: modelo, parâmetros, custo e parsing."""
from review_intel.core.prompts import SYSTEM, user_message
from review_intel.core.schema import json_schema, parse_output

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # US$ por token (até 100K tokens de prompt)
BATCH_DISCOUNT = 0.5
MAX_TOKENS = 2048  # inclui o raciocínio adaptativo, que no effort low é curto


def request_params(text: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        # o system prompt (~370 tokens) fica abaixo do mínimo cacheável, então não há cache_control
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_message(text)}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": json_schema()}},
    }


def cost(usage: dict, batch: bool = False) -> float:
    inp = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) * 1.25
    inp += usage.get("cache_read_input_tokens", 0) * 0.1
    c = inp * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT
    return c * (BATCH_DISCOUNT if batch else 1.0)


def record_from_message(review_id: str, message, batch: bool) -> dict:
    """Converte a resposta da API num registro do JSONL (analysis=None se recusou ou veio inválida)."""
    usage = {k: getattr(message.usage, k, 0) or 0 for k in
             ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    text = next((b.text for b in message.content if b.type == "text"), "")
    analysis = parse_output(text) if message.stop_reason == "end_turn" else None
    return {
        "review_id": review_id,
        "analysis": analysis.model_dump() if analysis else None,
        "stop_reason": message.stop_reason,
        "usage": usage,
        "cost_usd": cost(usage, batch),
        "model": MODEL,
    }
