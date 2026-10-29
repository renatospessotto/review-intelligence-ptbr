import pandas as pd
import pytest

from review_intel.evaluation.cascade import calibration, route, sweep
from review_intel.labeling.kappa import consolidate
from review_intel.core.schema import ASPECTS
from tests.conftest import make_analysis


def pred(label, conf, cost):
    return {"analysis": make_analysis(**{a: label for a in ASPECTS}), "confidence": {a: conf for a in ASPECTS},
            "cost_usd": cost}


def test_route_escalates_low_confidence():
    cheap = [pred("positivo", 0.9, 0.0), pred("positivo", 0.4, 0.0)]
    fb = [pred("negativo", 1.0, 0.01), pred("negativo", 1.0, 0.01)]
    chosen, esc = route(cheap, fb, threshold=0.5)
    assert list(esc) == [False, True]
    assert chosen[0] is cheap[0] and chosen[1] is fb[1]


def test_sweep_extremes():
    gold = [{a: "negativo" for a in ASPECTS}] * 2
    cheap = [pred("positivo", 0.9, 0.001), pred("negativo", 0.6, 0.001)]
    fb = [pred("negativo", 1.0, 0.01), pred("negativo", 1.0, 0.01)]
    t = sweep(cheap, fb, gold, [0.0, 0.7, 1.0]).set_index("limiar")
    assert t.loc[0.0, "% escalado"] == 0 and t.loc[1.0, "% escalado"] == 1
    assert t.loc[1.0, "F1 macro"] == pytest.approx(1.0)
    assert t.loc[0.0, "US$ / mil"] < t.loc[1.0, "US$ / mil"]


def test_calibration_perfect():
    gold = [{a: "positivo" for a in ASPECTS}]
    assert calibration([pred("positivo", 1.0, 0)], gold) == pytest.approx(0.0)


def test_gold_consolidation():
    def ann(lbl):
        return pd.DataFrame([{"review_id": "r1", "text": "t", **{a: "positivo" for a in ASPECTS},
                              "motivo_principal": "", "acao_sugerida": ""},
                             {"review_id": "r2", "text": "t", **{a: lbl for a in ASPECTS},
                              "motivo_principal": "", "acao_sugerida": ""}]).set_index("review_id")
    a, b = ann("positivo"), ann("negativo")
    gold, conflicts = consolidate(a, b)
    assert [g["review_id"] for g in gold] == ["r1"] and conflicts == ["r2"]
    adj = pd.DataFrame([{"review_id": "r2", **{f"final_{x}": "neutro" for x in ASPECTS}},
                        {"review_id": "r1", **{f"final_{x}": "" for x in ASPECTS}}]).set_index("review_id")
    gold, conflicts = consolidate(a, b, adj)
    assert len(gold) == 2 and not conflicts
