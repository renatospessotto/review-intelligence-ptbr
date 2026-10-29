---
language: [pt]
license: cc-by-nc-sa-4.0
base_model: neuralmind/bert-base-portuguese-cased
library_name: transformers
pipeline_tag: text-classification
tags: [bertimbau, aspect-based-sentiment-analysis, calibration, distillation, portuguese, e-commerce]
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
metrics: [f1]
---

# BERTimbau multi-aspecto para reviews em PT-BR

> **Projeto experimental.** Este modelo é o baseline clássico de um experimento que testou o **Laya**, um encoder de decisão
> novo, em português. Ele acabou empatado com o Laya, sendo o menor, o mais rápido e o mais bem calibrado.
> Código e relatório: [review-intelligence-ptbr](https://github.com/renatospessotto/review-intelligence-ptbr).

Um BERTimbau base com **uma cabeça de 4 classes por aspecto** (**entrega, produto, atendimento, preço, embalagem**:
`positivo`, `negativo`, `neutro` ou `ausente`) e temperature scaling por aspecto.

## Resultados

| | **BERTimbau ajustado** | Laya ajustado | Qwen3.5-2B + LoRA | Claude Haiku 5.5 (professor) |
|---|---|---|---|---|
| F1 macro no ouro (329 reviews) | **0,751** | 0,744 | 0,740 | 0,791 |
| ECE (calibração) | **0,006** | 0,014 | - | - |
| Latência p50 (Mac M4) | **5 ms** | 66 ms | 2,6 s | 1,7 s |
| US$ por mil reviews | **0,0006** | 0,006 | 0,27 | 0,18 |

- **O melhor no ouro, por margem pequena** (as diferenças entre 0,740 e 0,751 estão dentro do ruído de 329 reviews).
  Também é o melhor em **preço** (F1 0,97).
- **O mais barato:** cerca de 300 vezes menos que o Claude por review.
- **Cascata:** BERTimbau primeiro, com o Claude quando a menor confiança fica abaixo de 0,75. Assim, 21% dos reviews vão à API e o F1 chega a **0,845**.

## Uso

O modelo é uma classe pequena em PyTorch. Basta copiar este trecho:

```python
import json, torch
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

path = snapshot_download("RenatoSpessotto/bertimbau-aspectos-reviews-ptbr")
cfg = json.load(open(f"{path}/config.json"))
ASPECTS, LABELS = cfg["aspects"], cfg["labels"]

class MultiAspect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg["base"]))
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({a: nn.Linear(self.encoder.config.hidden_size, len(LABELS)) for a in ASPECTS})
        self.register_buffer("temperature", torch.ones(len(ASPECTS)))
    def forward(self, input_ids, attention_mask, **_):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return torch.stack([self.heads[a](cls) for a in ASPECTS], 1) / self.temperature.view(1, -1, 1)

model = MultiAspect(); model.load_state_dict(load_file(f"{path}/model.safetensors")); model.eval()
tok = AutoTokenizer.from_pretrained(path)

enc = tok(["Produto ótimo, mas chegou atrasado e a caixa amassada."], return_tensors="pt")
probs = model(**enc).softmax(-1)[0]
for a, p in zip(ASPECTS, probs):
    print(a, LABELS[p.argmax()], round(p.max().item(), 2))
```

## Treino

- **Base:** `neuralmind/bert-base-portuguese-cased` (110M).
- **Dados:** 9.000 reviews da Olist rotulados pelo Claude Haiku 5.5 (destilação), mais 1.000 de validação (acurácia por decisão na validação: 0,946).
- **Hiperparâmetros:** 3 épocas, lr 2e-5, batch 16, max_length 256.
- **Calibração:** temperatura por aspecto, ajustada na validação.
- **Avaliação:** conjunto ouro de 329 reviews, consenso entre Argon e Claude Opus 5.5 (kappa 0,86). Dataset: [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr).

## Limitações

- O ouro foi anotado por LLMs, não por pessoas.
- Os aspectos são fixos: um aspecto novo exige uma cabeça nova e retreino.
- O domínio é o de reviews de e-commerce da Olist (2016-2018).
- Licença: os dados de treino derivam do dataset da Olist (CC BY-NC-SA 4.0), por isso o modelo é para uso não comercial. O BERTimbau base é MIT.
