"""Análises para o relatório (notebooks/02_resultados.ipynb): carregar predições salvas, calibração,
risco × cobertura e confiança em reviews ambíguos. Nada aqui roda modelo; tudo parte de outputs/eval/."""
from pathlib import Path

import numpy as np
import pandas as pd

from review_intel.core.io import read_jsonl
from review_intel.core.schema import ASPECTS, aspect_labels


def load_predictions(pred_dir: str | Path, name: str, ids: list[str]) -> list[dict] | None:
    """Predições salvas de uma abordagem, na ordem de `ids`; None se faltar algum review."""
    path = Path(pred_dir) / f"{name}.jsonl"
    saved = {r["review_id"]: r for r in read_jsonl(path)}
    if not set(ids) <= set(saved):
        return None
    return [saved[i] for i in ids]


def decisions(preds: list[dict], gold: list[dict]) -> pd.DataFrame:
    """Uma linha por (review, aspecto): confiança da predição e se acertou o ouro."""
    rows = []
    for p, g in zip(preds, gold):
        labels = aspect_labels(p["analysis"])
        for a in ASPECTS:
            rows.append({"aspecto": a, "confidence": p["confidence"].get(a, np.nan) if p["confidence"] else np.nan,
                         "pred": labels.get(a) if labels else None, "gold": g[a], "correct": labels.get(a) == g[a]})
    return pd.DataFrame(rows)


def reliability_bins(conf, correct, n_bins: int = 10) -> pd.DataFrame:
    """Confiança média × acurácia por faixa de confiança (diagrama de confiabilidade)."""
    df = pd.DataFrame({"conf": np.asarray(conf, float), "correct": np.asarray(correct, float)})
    df["bin"] = np.clip((df["conf"] * n_bins).astype(int), 0, n_bins - 1)
    out = df.groupby("bin").agg(confianca=("conf", "mean"), acuracia=("correct", "mean"), n=("conf", "size"))
    return out.reset_index()


def risk_coverage(conf, correct, points: int = 20) -> pd.DataFrame:
    """Acurácia quando o modelo só responde as decisões mais confiantes (cobertura de 5% a 100%)."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    corr = np.asarray(correct, float)[order]
    cum = np.cumsum(corr) / np.arange(1, len(corr) + 1)
    cov = np.linspace(1 / points, 1, points)
    idx = np.maximum((cov * len(corr)).astype(int) - 1, 0)
    return pd.DataFrame({"cobertura": cov, "acuracia": cum[idx]})


def min_confidence(preds: list[dict]) -> np.ndarray:
    """Menor confiança entre os 5 aspectos de cada review (o mesmo critério da cascata)."""
    return np.array([min(p["confidence"].values()) for p in preds])
