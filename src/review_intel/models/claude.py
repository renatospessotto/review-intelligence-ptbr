"""LLM grande (teto da comparação): mesmo request da rotulagem prata, chamado de forma síncrona."""
from review_intel.core.claude_api import MODEL, record_from_message, request_params
from review_intel.core.predictors import Prediction, Predictor


class ClaudePredictor(Predictor):
    def __init__(self, client=None, name: str | None = None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.name = name or MODEL

    def predict(self, texts):
        out = []
        for t in texts:
            rec = record_from_message("", self.client.messages.create(**request_params(t)), batch=False)
            out.append(Prediction(rec["analysis"], cost_usd=rec["cost_usd"]))
        return out
