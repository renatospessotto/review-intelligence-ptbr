import pytest

from review_intel.core.metrics import aspect_f1, ece, kappa_per_aspect, latency_stats
from review_intel.core.schema import ASPECTS


def rows(labels):
    return [{a: lab for a in ASPECTS} for lab in labels]


def test_f1_perfect_and_invalid_counts_as_error():
    gold = rows(["positivo", "negativo", "neutro", "ausente"])
    assert aspect_f1(gold, gold)["macro"] == pytest.approx(1.0)
    pred = [None, *gold[1:]]  # 1 JSON inválido
    f1 = aspect_f1(gold, pred)["macro"]
    assert 0 < f1 < 1


def test_kappa_known_values():
    a = rows(["positivo", "negativo", "positivo", "negativo"])
    assert kappa_per_aspect(a, a)["media"] == pytest.approx(1.0)
    b = rows(["negativo", "positivo", "negativo", "positivo"])
    assert kappa_per_aspect(a, b)["media"] == pytest.approx(-1.0)
    same_single = rows(["ausente"] * 4)
    assert kappa_per_aspect(same_single, same_single)["media"] == 1.0


def test_ece():
    assert ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert ece([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)
    assert ece([], []) == 0.0


def test_latency_stats_ms():
    s = latency_stats([0.1] * 10)
    assert s["p50_ms"] == pytest.approx(100.0) and s["p95_ms"] == pytest.approx(100.0)


def test_f1_ignores_labels_absent_from_gold_and_pred():
    gold = rows(["negativo", "negativo"])
    assert aspect_f1(gold, gold)["macro"] == pytest.approx(1.0)
