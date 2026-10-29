import json
from types import SimpleNamespace

import pandas as pd

from review_intel.labeling import silver
from review_intel.core.io import read_jsonl
from tests.conftest import make_analysis


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def create(self, **params):
        self.calls.append(params)
        text, stop = self.replies[len(self.calls) - 1]
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=400, output_tokens=100, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0),
        )


def test_request_params_use_structured_output():
    p = silver.request_params("bom")
    assert p["model"] == "claude-haiku-5-5"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in p and "thinking" not in p


def test_cost():
    assert silver.cost({"input_tokens": 1_000_000, "output_tokens": 0}) == 0.10
    assert silver.cost({"input_tokens": 0, "output_tokens": 1_000_000}, batch=True) == 0.25


def test_label_sync_with_fake_client_and_resume(tmp_path):
    out = tmp_path / "silver.jsonl"
    df = pd.DataFrame({"review_id": ["a", "b", "c"], "text": ["chegou rápido", "x", "y"]})
    msgs = FakeMessages([(json.dumps(make_analysis(entrega="positivo")), "end_turn"),
                         ("não é json", "end_turn"),
                         ("", "refusal")])
    records = silver.label_sync(SimpleNamespace(messages=msgs), df, workers=1, out=out)
    by_id = {r["review_id"]: r for r in read_jsonl(out)}
    assert by_id["a"]["analysis"]["aspectos"]["entrega"] == "positivo"
    assert by_id["b"]["analysis"] is None and by_id["c"]["analysis"] is None
    assert by_id["c"]["stop_reason"] == "refusal"
    assert all(r["cost_usd"] > 0 for r in records)
    # retomada: só os inválidos voltam para a fila
    assert set(silver.pending(df, out=out)["review_id"]) == {"b", "c"}


def test_estimate_scales_with_batch_discount():
    df = pd.DataFrame({"review_id": ["a"] * 100, "text": ["texto de teste"] * 100})
    assert silver.estimate(df, batch=True)["usd"] < silver.estimate(df, batch=False)["usd"]


def test_gold_llm_with_fake_client():
    from review_intel.labeling import gold_llm

    labels = {"entrega": "negativo", "produto": "positivo", "atendimento": "ausente", "preco": "ausente",
              "embalagem": "ausente"}
    msgs = FakeMessages([(json.dumps(labels), "end_turn"), ("", "refusal")])
    df = pd.DataFrame({"review_id": ["a", "b"], "text": ["atrasou mas é bom", "x"]})
    out = gold_llm.run(SimpleNamespace(messages=msgs), df, "claude-opus-5-5", "medium", workers=1)
    assert list(out.columns) == ["review_id", "text", "entrega", "produto", "atendimento", "preco", "embalagem"]
    assert out.set_index("review_id").at["a", "entrega"] == "negativo"
    assert out.set_index("review_id").at["b", "entrega"] == ""  # recusa fica sem rótulo, sem fallback
    p = msgs.calls[0]
    assert "<guia>" in p["system"] and p["output_config"]["format"]["schema"]["required"] == list(gold_llm.ASPECTS)
