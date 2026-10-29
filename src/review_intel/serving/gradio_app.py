"""Demo: Laya e BERTimbau lado a lado, com a confiança por aspecto e a decisão da cascata.

    python -m review_intel.serving.gradio_app          # http://localhost:7860
"""
import time

import gradio as gr
import pandas as pd

from review_intel.core.schema import ASPECTS, aspect_labels
from review_intel.models.registry import load

THRESHOLD = 0.75  # abaixo disso, a cascata mandaria o review ao Claude
EXEMPLOS = [
    "Produto ótimo, mas chegou com 10 dias de atraso e a caixa toda amassada.",
    "Ainda não recebi meu pedido e ninguém responde no chat.",
    "Muito barato pela qualidade, recomendo!",
    "Chegou antes do prazo, bem embalado, mas o tamanho não serve.",
]
MODELS = {"Laya ajustado": "laya-ft", "BERTimbau": "bertimbau"}
_cache = {}


def model(name):
    if name not in _cache:
        _cache[name] = load(MODELS[name])
    return _cache[name]


def analyze(text: str):
    rows, notes = {a: {"aspecto": a} for a in ASPECTS}, []
    for name in MODELS:
        m = model(name)  # carrega (e baixa do Hub) fora da medição de tempo
        t0 = time.perf_counter()
        pred = m.predict([text])[0]
        ms = 1000 * (time.perf_counter() - t0)
        labels = aspect_labels(pred.analysis)
        for a in ASPECTS:
            rows[a][name] = f"{labels[a]} ({pred.confidence[a]:.0%})"
        low = min(pred.confidence.values())
        decision = "decide sozinho" if low >= THRESHOLD else "enviaria ao Claude"
        notes.append(f"**{name}**: menor confiança {low:.0%} → {decision} · {ms:.0f} ms")
    return pd.DataFrame(rows.values()), "\n\n".join(notes)


with gr.Blocks(title="Review Intelligence PT-BR") as demo:
    gr.Markdown("# Review Intelligence PT-BR\nCole uma avaliação de compra. Laya e BERTimbau classificam 5 aspectos "
                f"e dizem o quanto confiam. Abaixo de {THRESHOLD:.0%}, a cascata mandaria o review ao Claude.")
    text = gr.Textbox(lines=4, label="Avaliação")
    btn = gr.Button("Analisar", variant="primary")
    table = gr.Dataframe(label="Rótulo (confiança) por aspecto")
    info = gr.Markdown()
    btn.click(analyze, text, [table, info])
    gr.Examples(EXEMPLOS, text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
