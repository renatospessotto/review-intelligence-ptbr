# Guia de anotação: conjunto ouro

São 400 reviews (`gold_candidates`), rotulados **de forma independente** por duas pessoas, A e B. Quem anota não vê a nota do review nem o rótulo da outra pessoa.

## O que rotular
Em cada linha da planilha, preencha as 5 colunas de aspecto com **um** destes rótulos:

| Rótulo | Quando usar |
|---|---|
| `positivo` | o cliente elogia ou fica satisfeito com o aspecto |
| `negativo` | o cliente reclama ou fica insatisfeito |
| `neutro` | o aspecto aparece, mas sem julgamento claro |
| `ausente` | o aspecto não aparece, nem de forma implícita |

## Aspectos

| Aspecto | Inclui | Não inclui |
|---|---|---|
| `entrega` | prazo, atraso, rastreio, entrega errada ou incompleta, não recebimento | estado da caixa (vai em `embalagem`) |
| `produto` | qualidade, defeito, se corresponde ao anúncio, funcionamento, elogio genérico ("adorei", "recomendo") | preço |
| `atendimento` | contato com a loja ou o vendedor, resposta, troca, solução de problemas | entregador (vai em `entrega`) |
| `preco` | preço, custo-benefício, frete caro ou barato | — |
| `embalagem` | caixa amassada, proteção, lacre, embalagem bonita | produto quebrado sem menção à caixa (vai em `produto`) |

## Casos de borda
- **"Chegou tudo certo":** entrega `positivo`. Produto fica `ausente`, a menos que o produto seja avaliado.
- **"Ainda não recebi":** entrega `negativo`. Produto `ausente`.
- **"Recebi o produto errado":** entrega `negativo` e produto `negativo`.
- **"Chegou antes do prazo":** entrega `positivo`.
- **"Chegou dia 12":** entrega `neutro`.
- **Ironia** ("parabéns pela demora"): conta o sentido real, `negativo`.
- **Recomenda a loja:** atendimento só se fala do atendimento. Elogio genérico à loja vai em produto `positivo`.
- **Na dúvida entre `neutro` e `ausente`:** se o aspecto é citado, `neutro`; se é só inferido, `ausente`.

## Colunas de texto (opcionais)
`motivo_principal` e `acao_sugerida`, com até 12 palavras cada. Não entram no kappa.

## Fluxo
1. `python -m labeling.gold_export` gera `labeling/out/gold_anotador_A.csv` e `gold_anotador_B.csv`, com as linhas em ordens diferentes.
2. Cada pessoa preenche a sua planilha. Pode abrir no Google Sheets e baixar como CSV.
3. `python -m labeling.kappa` calcula o kappa por aspecto e grava as divergências em `gold_conflitos.csv`.
4. As duas pessoas resolvem os conflitos juntas e preenchem a coluna `final_<aspecto>`.
5. Rodar `python -m labeling.kappa` de novo grava `gold.jsonl`, o ouro consolidado.
