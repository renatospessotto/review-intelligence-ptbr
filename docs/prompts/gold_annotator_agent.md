# Prompt do anotador Opus 5.5 (execução original)

O segundo anotador do conjunto ouro foi o **Claude Opus 5.5**, rodando como subagente do Claude Code em 28/10/2026. O subagente não tinha contexto da conversa e recebeu só o prompt abaixo. Ele leu o guia (`labeling/guia_anotacao.md`) e os 400 textos, e anotou review por review, em blocos de 50 a 95. Não chamou API de modelo e não usou regras automáticas.

Para reproduzir sem o Claude Code, use `python -m review_intel.labeling.gold_llm`. O script manda o mesmo guia e a mesma tarefa pela API da Anthropic.

---

Você é um anotador humano-equivalente criando um conjunto de avaliação ("ouro") para extração de aspectos em avaliações de compras online em português do Brasil. Faça a anotação você mesmo, lendo cada texto. NÃO escreva código que classifique por palavras-chave, regras ou heurísticas, e NÃO chame nenhuma API de modelo: o valor deste trabalho é o seu julgamento, review por review.

## Arquivos
- Guia de anotação (LEIA INTEIRO ANTES DE COMEÇAR): `labeling/guia_anotacao.md`
- Entrada: `labeling/out/gold_anotador_A.csv` (planilha em branco), colunas `review_id,text,...`. Use só `review_id` e `text`. São 400 linhas.
- Saída: `labeling/out/gold_opus.csv`

## Independência (obrigatório)
Não abra, leia, liste conteúdo nem use de nenhuma forma os arquivos com rótulos de outros anotadores: as planilhas preenchidas pelo Argon, `labeling/out/silver.jsonl`, `labeling/out/smoke.jsonl`, nem qualquer coisa em `outputs/`.

## Tarefa
Para cada um dos 400 reviews, atribua a cada um dos 5 aspectos exatamente um rótulo:
`positivo`, `negativo`, `neutro` ou `ausente`.
Aspectos: `entrega`, `produto`, `atendimento`, `preco`, `embalagem`.
Siga as definições, a tabela "Inclui / Não inclui" e os casos de borda do guia. Resumo:
- `ausente`: o aspecto não aparece no texto, nem implicitamente (a maioria das células será `ausente`).
- `neutro`: o aspecto é citado sem julgamento claro.
- Elogio genérico ("ótimo", "recomendo", "adorei") sem dizer do quê → produto `positivo`.
- Produto errado recebido → entrega `negativo` e produto `negativo`.
- Não recebeu → entrega `negativo`, produto `ausente`.
- Produto quebrado sem mencionar a caixa → produto (não embalagem).
- Ironia: rotule o sentido real.

## Formato da saída
CSV com cabeçalho exatamente `review_id,entrega,produto,atendimento,preco,embalagem`. Uma linha por review_id da entrada (todos os 400, nenhum a mais), rótulos em minúsculas.

## Como trabalhar
1. Leia o guia.
2. Leia os textos em blocos (por exemplo, 50 por vez) e, para cada bloco, escreva os seus rótulos no CSV de saída.
3. No fim, valide: 400 linhas, conjunto de review_id idêntico ao da entrada, sem duplicatas, todas as células em {positivo, negativo, neutro, ausente}.
