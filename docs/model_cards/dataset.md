---
language: [pt]
license: cc-by-nc-sa-4.0
task_categories: [text-classification]
tags: [aspect-based-sentiment-analysis, e-commerce, olist, llm-annotation, distillation, portuguese]
size_categories: [10K<n<100K]
configs:
- config_name: gold
  data_files:
  - split: test
    path: gold/test.jsonl
- config_name: silver
  data_files:
  - split: train
    path: silver/train.jsonl
  - split: validation
    path: silver/validation.jsonl
---

# Aspectos em reviews da Olist (PT-BR): ouro de consenso e rótulos do Claude

> **Projeto experimental.** Estes dados foram montados para testar o **Laya**, um encoder de decisão novo, em português,
> contra o BERTimbau e um LLM pequeno ajustado. Código e relatório: [review-intelligence-ptbr](https://github.com/renatospessotto/review-intelligence-ptbr).

Reviews de compras online da [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Cada review tem o sentimento de 5 aspectos: **entrega, produto, atendimento, preço e embalagem**. Os rótulos possíveis são `positivo`, `negativo`, `neutro` e `ausente`.

| Config | Split | Reviews | Rótulos |
|---|---|---|---|
| `gold` | test | 329 | consenso de dois anotadores LLM independentes e às cegas (Argon e Claude Opus 5.5). Ficam só os reviews em que os dois concordam nos 5 aspectos (kappa médio 0,86). |
| `silver` | train / validation | 9.000 / 1.000 | Claude Haiku 5.5 com saída estruturada. Inclui `motivo_principal` e `acao_sugerida`. |

```python
from datasets import load_dataset
gold = load_dataset("RenatoSpessotto/olist-aspectos-reviews-ptbr", "gold", split="test")
silver = load_dataset("RenatoSpessotto/olist-aspectos-reviews-ptbr", "silver")
```

## Concordância no ouro (400 reviews anotados, 329 de consenso)

| Aspecto | Kappa | Concordância |
|---|---|---|
| entrega | 0,900 | 93,3% |
| produto | 0,931 | 95,8% |
| atendimento | 0,725 | 92,8% |
| preço | 0,855 | 98,5% |
| embalagem | 0,888 | 99,0% |

As duas anotações originais, o kappa e os 71 reviews sem consenso estão em `gold/annotations/`, `gold/kappa.json` e `gold/descartados.csv`.

## Campos

- `review_id` e `text`, que junta título e mensagem.
- `review_score`, a nota de 1 a 5.
- `entrega`, `produto`, `atendimento`, `preco` e `embalagem`.
- Só no `silver`: `motivo_principal` e `acao_sugerida`.

## Limitações

- **Anotação por LLMs.** Nenhum dos conjuntos foi anotado por pessoas.
- **Viés de família.** Um dos anotadores do ouro é da mesma família do modelo que gerou o prata.
- **Viés para casos fáceis.** O consenso descarta os casos ambíguos.
- **Classes raras.** Preço, embalagem e `neutro` têm poucos exemplos.

## Licença

CC BY-NC-SA 4.0, herdada do dataset da Olist. Uso não comercial, com atribuição e compartilhamento pela mesma licença.
