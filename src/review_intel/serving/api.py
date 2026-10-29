"""API de extração: POST /extract {"text": "..."} → JSON do esquema.

    uvicorn review_intel.serving.api:app                         # Laya ajustado (padrão)
    RI_APPROACH=bertimbau uvicorn review_intel.serving.api:app

RI_APPROACH aceita as abordagens de review_intel.models.registry. Os modelos vêm de outputs/ se existirem,
senão do Hugging Face.
"""
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from review_intel.core.predictors import Predictor

app = FastAPI(title="Review Intelligence PT-BR")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class ExtractResponse(BaseModel):
    approach: str
    analysis: dict | None
    confidence: dict[str, float]
    latency_ms: float


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    from review_intel.core.env import load_dotenv
    from review_intel.models.registry import load

    load_dotenv()
    return load(os.environ.get("RI_APPROACH", "laya-ft"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    predictor = get_predictor()
    t0 = time.perf_counter()
    pred = predictor.predict([req.text])[0]
    if pred.analysis is None:
        raise HTTPException(status_code=502, detail="o modelo não gerou um JSON válido")
    return ExtractResponse(approach=predictor.name, analysis=pred.analysis, confidence=pred.confidence,
                           latency_ms=1000 * (time.perf_counter() - t0))
