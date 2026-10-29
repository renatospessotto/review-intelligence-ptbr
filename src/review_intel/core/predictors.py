"""Interface comum: toda abordagem recebe textos e devolve um JSON do esquema (ou None)."""
import time
from dataclasses import dataclass, field

from review_intel.core.schema import ReviewAnalysis


@dataclass
class Prediction:
    analysis: dict | None  # ReviewAnalysis.model_dump() ou None se a saída for inválida
    confidence: dict[str, float] = field(default_factory=dict)  # por aspecto, quando o modelo dá probabilidade
    cost_usd: float = 0.0


class Predictor:
    name = "base"

    def predict(self, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError

    def timed_predict(self, texts: list[str], batch_size: int = 8):
        """Prediz em lotes e devolve (predições, segundos por item)."""
        preds, secs = [], []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()
            preds.extend(self.predict(chunk))
            secs.extend([(time.perf_counter() - t0) / len(chunk)] * len(chunk))
        return preds, secs


def as_dict(analysis: ReviewAnalysis | None) -> dict | None:
    return analysis.model_dump() if analysis is not None else None
