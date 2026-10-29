import json

from review_intel.core.prompts import SYSTEM
from review_intel.core.schema import ASPECTS, LABELS, aspect_labels, json_schema, parse_output
from tests.conftest import make_analysis


def test_parse_valid_json():
    a = make_analysis(entrega="negativo")
    out = parse_output(json.dumps(a))
    assert out is not None and out.aspectos.entrega == "negativo"


def test_parse_fenced_and_surrounding_text():
    a = make_analysis(produto="positivo")
    assert parse_output(f"Claro!\n```json\n{json.dumps(a)}\n```\nPronto.") is not None
    assert parse_output(f"<think>hmm {{}}</think>{json.dumps(a)}") is not None


def test_parse_rejects_invalid():
    a = make_analysis()
    assert parse_output("") is None
    assert parse_output("sem json aqui") is None
    assert parse_output(json.dumps(a)[:-5]) is None  # truncado
    bad = make_analysis(entrega="ótimo")  # rótulo fora do conjunto
    assert parse_output(json.dumps(bad)) is None
    extra = dict(a, extra=1)
    assert parse_output(json.dumps(extra)) is None
    missing = {"aspectos": {"entrega": "positivo"}, "motivo_principal": "", "acao_sugerida": ""}
    assert parse_output(json.dumps(missing)) is None


def test_aspect_labels_none_for_invalid():
    assert aspect_labels(None) == {a: None for a in ASPECTS}


def test_json_schema_matches_constants():
    s = json_schema()
    assert s["additionalProperties"] is False
    assert set(s["properties"]["aspectos"]["required"]) == set(ASPECTS)
    assert s["properties"]["aspectos"]["properties"]["entrega"]["enum"] == list(LABELS)


def test_prompt_example_is_valid_output():
    example = SYSTEM[SYSTEM.rfind("{\"aspectos\""):]
    assert parse_output(example) is not None
