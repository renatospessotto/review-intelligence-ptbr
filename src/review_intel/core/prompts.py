"""Prompt único usado pelo rotulador (Claude) e pelos LLMs pequenos, para a comparação ser justa."""
from review_intel.core.schema import ASPECTS

SYSTEM = """Você analisa avaliações de compras online em português do Brasil.
Para cada aspecto, diga se o cliente fala dele de forma positiva, negativa ou neutra, ou se não fala dele (ausente).

Aspectos:
- entrega: prazo, atraso, rastreio, entrega errada ou incompleta, não recebimento
- produto: qualidade, defeito, se corresponde ao anúncio, funcionamento
- atendimento: contato com a loja/vendedor, resposta, solução de problemas
- preco: preço, custo-benefício, frete caro ou barato
- embalagem: estado da embalagem, proteção, caixa danificada

Regras:
- "ausente" quando o texto não menciona o aspecto, nem implicitamente.
- "neutro" quando menciona sem julgamento claro (ex.: "chegou dia 10").
- Elogio genérico ("adorei", "recomendo") conta como produto positivo.
- motivo_principal: até 12 palavras com a razão central da nota.
- acao_sugerida: até 12 palavras com o que a loja deveria fazer ("nenhuma" se nada).

Responda só com o JSON, com exatamente estas chaves e um dos 4 rótulos por aspecto. Exemplo para "Produto bom, mas veio com a caixa amassada e demorou demais":
{"aspectos": {"entrega": "negativo", "produto": "positivo", "atendimento": "ausente", "preco": "ausente", "embalagem": "negativo"}, "motivo_principal": "entrega demorada e caixa amassada", "acao_sugerida": "melhorar prazo e proteção da embalagem"}"""

assert all(a in SYSTEM for a in ASPECTS)


def user_message(text: str) -> str:
    return f"Avaliação:\n{text.strip()}"


def chat_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(text)}]
