# Imagem única para a API e a demo. Os modelos são baixados do Hugging Face na primeira chamada
# (ou lidos de ./outputs, se montado). Uso: docker compose up --build
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 RI_ROOT=/app HF_HOME=/cache/hf
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir .
EXPOSE 8000 7860
CMD ["uvicorn", "review_intel.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
