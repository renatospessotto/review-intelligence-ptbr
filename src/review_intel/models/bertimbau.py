"""BERTimbau com uma cabeça de 4 classes por aspecto (baseline clássico) e temperature scaling por aspecto."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from review_intel.core.device import best_device
from review_intel.core.predictors import Prediction, Predictor
from review_intel.core.schema import ASPECTS, LABELS

BASE = "neuralmind/bert-base-portuguese-cased"
L2I = {label: i for i, label in enumerate(LABELS)}


class MultiAspectClassifier(nn.Module):
    def __init__(self, base: str = BASE, pretrained: bool = True):
        super().__init__()
        # ao carregar um modelo já treinado, os pesos do encoder vêm do checkpoint: basta a arquitetura
        self.encoder = AutoModel.from_pretrained(base) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(base))
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({a: nn.Linear(h, len(LABELS)) for a in ASPECTS})
        self.register_buffer("temperature", torch.ones(len(ASPECTS)))

    def forward(self, input_ids, attention_mask, **_) -> torch.Tensor:
        """Logits [batch, n_aspectos, n_rótulos], já divididos pela temperatura."""
        cls = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
        logits = torch.stack([self.heads[a](cls) for a in ASPECTS], dim=1)
        return logits / self.temperature.view(1, -1, 1)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "config.json").write_text(json.dumps({"base": self.encoder.name_or_path, "aspects": ASPECTS,
                                                      "labels": LABELS}))

    @classmethod
    def load(cls, path):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        model = cls(cfg["base"], pretrained=False)
        if (path / "model.safetensors").exists():  # formato publicado no Hugging Face
            from safetensors.torch import load_file

            state = load_file(path / "model.safetensors")
        else:
            state = torch.load(path / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        model.base = cfg["base"]
        return model


class BertimbauPredictor(Predictor):
    def __init__(self, path, device: str | None = None, name: str = "BERTimbau multi-aspecto"):
        self.device = device or best_device()
        self.model = MultiAspectClassifier.load(path).to(self.device).eval()
        has_tok = (Path(path) / "tokenizer_config.json").exists()
        self.tok = AutoTokenizer.from_pretrained(path if has_tok else self.model.base)
        self.name = name

    @torch.inference_mode()
    def predict(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)
        probs = self.model(**enc).softmax(-1).cpu()
        preds = []
        for p in probs:
            conf, idx = p.max(-1)
            preds.append(Prediction(
                {"aspectos": {a: LABELS[i] for a, i in zip(ASPECTS, idx.tolist())}, "motivo_principal": "",
                 "acao_sugerida": ""},
                confidence={a: float(c) for a, c in zip(ASPECTS, conf.tolist())},
            ))
        return preds
