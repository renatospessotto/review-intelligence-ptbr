# Conjunto ouro: aspectos em avaliações do Olist (PT-BR)

Conjunto de avaliação com **329 avaliações de compras online em português do Brasil**. Cada uma tem um rótulo de sentimento para 5 aspectos: entrega, produto, atendimento, preço e embalagem.

Ele serve para medir os modelos do projeto e **nunca** foi usado em treino.

## Origem dos textos
- **Fonte:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), arquivo `olist_order_reviews_dataset.csv`. Licença **CC BY-NC-SA 4.0**, que este conjunto herda: uso não comercial, com atribuição e compartilhamento pela mesma licença.
- **Amostra:** 400 reviews de `gold_candidates`, sorteados com estratificação pela nota (semente 42, `data/prepare.py`).
  - Os 400 são disjuntos dos 10.000 reviews usados no treino (`silver_pool`).
  - O texto é título + mensagem, com quebras de linha removidas.
  - Duplicatas exatas foram removidas antes do sorteio.

## Esquema
Cada aspecto recebe um rótulo:

| Rótulo | Significado |
|---|---|
| `positivo` | o cliente elogia o aspecto |
| `negativo` | o cliente reclama do aspecto |
| `neutro` | o aspecto aparece sem julgamento |
| `ausente` | o aspecto não aparece |

As definições e os casos de borda estão em [`docs/guia_anotacao.md`](../../docs/guia_anotacao.md).

## Como foi anotado
O conjunto foi anotado por **dois LLMs, de forma independente e às cegas**. Nenhum viu os rótulos do outro, os rótulos de treino ou a nota do review.

| Anotador | Como | Arquivo |
|---|---|---|
| **Argon** | anotação feita pelo autor do projeto com o Argon. *A completar: nome completo e versão do modelo.* | `annotations/argon.csv` |
| **Claude Opus 5.5** | subagente do Claude Code sem contexto da conversa, que recebeu só o guia e os textos (28/10/2026). O prompt exato está em [`docs/prompts/gold_annotator_agent.md`](../../docs/prompts/gold_annotator_agent.md) | `annotations/opus-5.5.csv` |

O **ouro de consenso** (`gold.jsonl`) fica só com os reviews em que **os dois anotadores concordam nos 5 aspectos**: 329 dos 400. Os 71 descartados, com os rótulos de cada anotador, estão em `gold_descartados.csv`.

### Concordância entre os anotadores (400 reviews)

| Aspecto | Kappa de Cohen | Concordância bruta |
|---|---|---|
| entrega | 0,900 | 93,3% |
| produto | 0,931 | 95,8% |
| atendimento | 0,725 | 92,8% |
| preço | 0,855 | 98,5% |
| embalagem | 0,888 | 99,0% |
| **média** | **0,860** | |

A maior divergência está em **atendimento**: o Argon marcou o aspecto em 72 reviews e o Opus em 46. Os dois leem de forma diferente o elogio ou a crítica genérica à "loja" ou ao "vendedor", que pelo guia conta como produto.

### Distribuição no ouro de consenso (329 reviews)

| Aspecto | positivo | negativo | neutro | ausente |
|---|---|---|---|---|
| entrega | 115 | 97 | 1 | 116 |
| produto | 178 | 48 | 8 | 95 |
| atendimento | 10 | 29 | 1 | 289 |
| preço | 11 | 5 | 0 | 313 |
| embalagem | 9 | 3 | 1 | 316 |

## Como reproduzir
```bash
make gold   # consenso + kappa a partir de annotations/*.csv; deve gerar gold.jsonl idêntico ao versionado

# refazer a anotação do Claude via API (guia como system prompt, um review por chamada; ~US$ 3,80 com Opus 5.5)
python -m review_intel.labeling.gold_llm --dry-run
python -m review_intel.labeling.gold_llm --model claude-opus-5-5 --out data/gold/annotations/opus-5.5-api.csv
```
O `gold_llm.py` é o equivalente por API do subagente original. Como LLMs não são determinísticos, uma nova execução pode divergir em alguns reviews. O arquivo versionado em `annotations/` é o que define o ouro.

## Limitações
- **Não é anotação humana.** O kappa mede a concordância entre dois LLMs, não entre pessoas.
- **Possível viés de família.** Os modelos ajustados no projeto foram treinados com rótulos do Claude Haiku 5.5, e um dos anotadores do ouro também é um Claude. Isso pode favorecer esses modelos. O Argon, de fora da família Claude, reduz esse efeito, e o consenso exige que os dois concordem.
- **O consenso tende aos casos fáceis.** Ao descartar os 71 reviews com divergência, saem justamente os ambíguos, e os números no ouro ficam um pouco otimistas. Os descartados continuam disponíveis para análise.
- **Classes raras.** Preço, embalagem e o rótulo `neutro` têm poucos exemplos, então o F1 desses casos tem variância alta.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `annotations/argon.csv` | 400 reviews: `review_id`, `text`, 5 aspectos, `motivo_principal`, `acao_sugerida` |
| `annotations/opus-5.5.csv` | 400 reviews: `review_id`, `text`, 5 aspectos |
| `gold.jsonl` | 329 reviews de consenso no formato do projeto (`review_id`, `analysis`) |
| `gold_descartados.csv` | 71 reviews sem consenso, com os rótulos de cada anotador |
| `kappa.json` | kappa e concordância bruta por aspecto |
