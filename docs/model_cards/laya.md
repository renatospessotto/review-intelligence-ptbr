---
language: [pt]
license: cc-by-nc-sa-4.0
base_model: convaiinnovations/laya
library_name: laya
pipeline_tag: text-classification
tags: [laya, aspect-based-sentiment-analysis, calibration, distillation, portuguese, e-commerce]
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
metrics: [f1]
---

# Laya ajustado para aspectos de reviews em PT-BR

> **Projeto experimental.** Este modelo existe para testar o **Laya** (Convai, set/2026), um encoder de decisão aberto
> com probabilidades calibradas, em português, numa tarefa real. Não é um produto: é o resultado de um experimento,
> com os números e as limitações abaixo. Código e relatório: [review-intelligence-ptbr](https://github.com/renatospessotto/review-intelligence-ptbr).

Dado um review de compra, o modelo responde 5 perguntas de múltipla escolha, uma por aspecto (**entrega, produto,
atendimento, preço, embalagem**), com `positivo`, `negativo`, `neutro` ou `ausente`, e uma **confiança calibrada**
para cada resposta.

## Resultados

| | Laya zero-shot | **Laya ajustado** | BERTimbau ajustado | Claude Haiku 5.5 (professor) |
|---|---|---|---|---|
| F1 macro no ouro (329 reviews) | 0,187 | **0,744** | 0,751 | 0,791 |
| ECE (calibração) | 0,530 | **0,014** | 0,006 | - |
| Latência p50 (Mac M4) | 135 ms | **66 ms** | 5 ms | 1,7 s |

- **O fine-tuning é o que faz o Laya funcionar em português.** Zero-shot, ele colapsa numa única resposta com ~80% de
  confiança. Depois do ajuste, na validação, a acurácia por decisão sobe de 0,28 para 0,94 e o ECE cai de 0,53 para 0,02.
- **Melhor dos modelos pequenos em atendimento (0,77) e embalagem (0,79).**
- **Confiança útil:** respondendo só os 80% de decisões mais confiantes, a acurácia vai de 95% para 99,3%. Nos reviews
  em que dois anotadores discordaram, a fração de reviews com confiança abaixo de 0,75 dobra (20% → 39%).
- **Cascata:** o Laya decide sozinho quando a menor confiança é ≥ 0,75 e manda o resto ao Claude. Assim, só 20% dos
  reviews vão à API e o F1 chega a **0,836** (contra 0,791 do Claude sozinho). O limiar foi escolhido no próprio ouro, então
  esse número é otimista.

## Uso

```python
import json, laya
from huggingface_hub import snapshot_download

path = snapshot_download("RenatoSpessotto/laya-aspectos-reviews-ptbr")
agent = laya.load(path)                      # device="cuda" | "mps" | "cpu"
questions = json.load(open(f"{path}/questions.json"))

out = agent.predict_batch(["Produto ótimo, mas chegou atrasado e a caixa amassada."], questions)[0]
for aspect, ans in out["answers"].items():
    print(aspect, ans["choice"], round(ans["answer_confidence"], 2))
```

As perguntas em `questions.json` são as do treino. Perguntas novas funcionam no formato, mas o modelo não foi treinado
nelas: num teste qualitativo, as respostas vieram com confiança baixa e inconsistentes.

## Treino

- **Base:** `convaiinnovations/laya`, subpasta `multilingual` (encoder mmBERT, ~322M).
- **Dados:** 9.000 reviews da Olist rotulados pelo Claude Haiku 5.5 (destilação), mais 1.000 de validação. São 44.600 itens de treino (5 perguntas por review).
- **Hiperparâmetros:** `laya.train.finetune`, 4 épocas, soft-CE, embaralhamento de opções, lr do encoder 2,5e-5, lr da cabeça 1e-4, batch efetivo 64, no Colab (GPU L4/A100).
- **Calibração:** temperatura ajustada em 400 itens separados (T = 2,38).
- **Avaliação:** conjunto ouro de 329 reviews, consenso de dois anotadores LLM independentes (Argon e Claude Opus 5.5), com kappa 0,86. Dataset: [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr).

## Limitações

- O ouro foi anotado por LLMs, não por pessoas, e um dos anotadores é da mesma família do professor.
- Preço e embalagem têm poucos exemplos no ouro (16 e 13), então o F1 desses aspectos varia muito.
- O domínio é o de reviews de e-commerce da Olist (2016-2018).
- Licença: os dados de treino derivam do dataset da Olist (CC BY-NC-SA 4.0), por isso o modelo segue a mesma licença, para uso não comercial. O Laya base é Apache-2.0.
