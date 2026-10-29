"""Métricas: F1 por aspecto, kappa, calibração (ECE) e latência."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from review_intel.core.schema import ASPECTS, LABELS

INVALID = "__invalido__"


def aspect_f1(gold: list[dict], pred: list[dict]) -> dict[str, float]:
    """F1-macro sobre as 4 classes, por aspecto, e a média entre aspectos.

    `gold`/`pred` são listas de {aspecto: rótulo}. Predição None (JSON inválido) conta como erro.
    """
    out = {}
    for a in ASPECTS:
        y = [g[a] for g in gold]
        p = [(q.get(a) if q else None) or INVALID for q in pred]
        # só rótulos que aparecem no ouro ou na predição: classe ausente dos dois não deve valer 0 na média
        present = [lab for lab in LABELS if lab in set(y) | set(p)]
        out[a] = f1_score(y, p, labels=present, average="macro", zero_division=0)
    out["macro"] = float(np.mean([out[a] for a in ASPECTS]))
    return out


def kappa_per_aspect(a: list[dict], b: list[dict]) -> dict[str, float]:
    out = {}
    for asp in ASPECTS:
        ya, yb = [x[asp] for x in a], [x[asp] for x in b]
        # kappa é indefinido quando os dois anotadores usam um único rótulo idêntico
        out[asp] = 1.0 if len(set(ya) | set(yb)) == 1 else float(cohen_kappa_score(ya, yb))
    out["media"] = float(np.mean([out[x] for x in ASPECTS]))
    return out


def ece(confidences, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error com bins de largura igual."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - corr[mask].mean())
    return float(total)


def latency_stats(seconds_per_item) -> dict[str, float]:
    s = np.asarray(seconds_per_item, dtype=float) * 1000
    return {"p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95))}
