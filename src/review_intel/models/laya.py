"""Laya (encoder de decisão): uma pergunta `choice` por aspecto, com probabilidades."""
from review_intel.core.device import best_device
from review_intel.core.predictors import Prediction, Predictor
from review_intel.core.schema import ASPECTS

ASPECT_DESC = {
    "entrega": "a entrega (prazo, atraso, rastreio, recebimento)",
    "produto": "o produto (qualidade, defeito, se corresponde ao anúncio)",
    "atendimento": "o atendimento da loja ou do vendedor",
    "preco": "o preço, o custo-benefício ou o frete",
    "embalagem": "a embalagem (estado da caixa, proteção)",
}


def questions() -> dict:
    return {
        a: {
            "type": "choice",
            "instructions": f"Como o cliente avalia {desc}?",
            "criteria": {
                "positivo": "elogia ou fica satisfeito",
                "negativo": "reclama ou fica insatisfeito",
                "neutro": "menciona sem julgamento",
                "ausente": "não fala disso",
            },
        }
        for a, desc in ASPECT_DESC.items()
    }


class LayaPredictor(Predictor):
    """`checkpoint` é o repo do Hub (zero-shot) ou uma pasta local gerada por laya.train.finetune."""

    def __init__(self, checkpoint: str = "convaiinnovations/laya", subfolder: str | None = "multilingual",
                 device: str | None = None, name: str | None = None):
        import laya

        kwargs = {"subfolder": subfolder} if subfolder else {}
        self.agent = laya.load(checkpoint, device=device or best_device(), **kwargs)
        self.q = questions()
        self.name = name or ("Laya zero-shot" if subfolder else "Laya fine-tuned")

    def predict(self, texts):
        results = self.agent.predict_batch(list(texts), self.q)
        preds = []
        for r in results:
            ans = r["answers"]
            preds.append(Prediction(
                # o Laya só decide; motivo e ação ficam vazios (os modelos generativos cuidam deles)
                {"aspectos": {a: ans[a]["choice"] for a in ASPECTS}, "motivo_principal": "", "acao_sugerida": ""},
                confidence={a: float(ans[a]["answer_confidence"]) for a in ASPECTS},
            ))
        return preds
