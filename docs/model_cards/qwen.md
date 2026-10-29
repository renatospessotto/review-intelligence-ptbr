---
language: [pt]
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: text-generation
tags: [lora, qlora, distillation, structured-output, aspect-based-sentiment-analysis, portuguese]
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
---

# Qwen3.5-2B + LoRA: extração de aspectos de reviews em PT-BR (JSON)

> **Projeto experimental.** Este adapter é a referência generativa de um experimento cujo foco foi testar o **Laya**
> (encoder de decisão) contra o BERTimbau. Código e relatório: [review-intelligence-ptbr](https://github.com/renatospessotto/review-intelligence-ptbr).

Adapter QLoRA que faz o Qwen3.5-2B devolver o mesmo JSON do professor (Claude Haiku 5.5): o rótulo de cada um dos 5 aspectos,
o `motivo_principal` e a `acao_sugerida`.

## Resultados

| | Qwen zero-shot | **Qwen + LoRA** | Laya ajustado | BERTimbau ajustado | Claude Haiku 5.5 |
|---|---|---|---|---|---|
| F1 macro no ouro | 0,648 | **0,740** | 0,744 | 0,751 | 0,791 |
| F1 na validação (rótulos do Haiku) | 0,581 | **0,751** | 0,737 | 0,693 | - |
| Latência p50 (Mac M4) | 3,4 s | 2,6 s | 66 ms | 5 ms | 1,7 s |

**Conclusão do experimento:**
- É o modelo que melhor imita o professor e o único dos pequenos que gera texto.
- Na classificação, empata com os encoders e é de 40 a 500 vezes mais lento.
- Não tem confiança nativa, então não serve como primeiro estágio de uma cascata.
- Só compensa quando o texto gerado é necessário.

## Uso

```python
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "RenatoSpessotto/qwen3.5-2b-aspectos-reviews-ptbr-lora"
tok = AutoTokenizer.from_pretrained(repo)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", torch_dtype="auto"), repo)
system = open(hf_hub_download(repo, "system_prompt.txt")).read()

msgs = [{"role": "system", "content": system},
        {"role": "user", "content": "Avaliação:\nProduto ótimo, mas chegou atrasado e a caixa amassada."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=200)[0, ids.shape[1]:], skip_special_tokens=True))
```

## Treino

- **Método:** QLoRA em 4 bits com TRL SFTTrainer, LoRA com r=16 em todas as camadas lineares e perda só na resposta (o JSON).
- **Hiperparâmetros:** 2 épocas, lr 2e-4, batch efetivo 16, no Colab (GPU L4/A100).
- **Dados:** 9.000 reviews da Olist rotulados pelo Claude Haiku 5.5.

## Limitações

- O ouro foi anotado por LLMs.
- A latência foi medida num Mac, sem os kernels otimizados de atenção linear.
- Licença: os dados de treino derivam do dataset da Olist (CC BY-NC-SA 4.0), por isso o adapter é para uso não comercial.
