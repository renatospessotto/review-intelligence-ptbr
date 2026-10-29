PY ?= .venv/bin/python
SMOKE_MODEL = Qwen/Qwen3.5-0.8B
M = $(PY) -m review_intel

.PHONY: help setup test notebook-tests docker api demo data silver gold train-colab eval eval-val cascade report publish publish-dry colab-bundle import-models smoke

help:            ## lista os comandos
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*## /\t/'

# --- usar ---
setup:           ## cria o venv e instala o pacote com tudo (treino, testes, notebooks)
	python3.11 -m venv .venv && $(PY) -m pip install -e ".[dev]"
	$(PY) -m ipykernel install --prefix .venv --name review-intel --display-name "review-intel (.venv)"
test:            ## testes unitários (sem GPU e sem API)
	$(PY) -m pytest -q
docker:          ## sobe API (:8000) e demo (:7860) com os modelos do Hugging Face
	docker compose up --build
api:             ## API local (RI_APPROACH=laya-ft|bertimbau|...)
	$(PY) -m uvicorn review_intel.serving.api:app --port 8000
demo:            ## demo Gradio local, Laya x BERTimbau
	$(PY) -m review_intel.serving.gradio_app

# --- reproduzir ---
data:            ## baixa o Olist (precisa de ~/.kaggle/kaggle.json) e gera os splits e as planilhas do ouro
	$(M).data.download && $(M).data.prepare && $(M).labeling.gold_export
silver:          ## rótulos prata com o Claude Haiku 5.5 pela Batch API (precisa de ANTHROPIC_API_KEY)
	$(M).labeling.silver --mode batch
gold:            ## reconstrói o ouro de consenso e o kappa a partir de data/gold/annotations/
	$(M).labeling.kappa --a data/gold/annotations/argon.csv --b data/gold/annotations/opus-5.5.csv \
		--name-a argon --name-b opus-5.5 --agreement-only --out-dir data/gold
colab-bundle:    ## empacota código e dados para treinar no Colab (notebooks/03_treino_colab.ipynb)
	$(M).hub.colab_bundle
import-models:   ## descompacta modelos treinados no Colab em outputs/ (ZIP=...)
	$(M).hub.import_models $(ZIP)
eval:            ## tabela comparativa no ouro (reaproveita as predições salvas)
	$(M).evaluation.run --from-predictions
eval-val:        ## tabela na validação prata (rótulos do Haiku)
	$(M).evaluation.run --silver-val --from-predictions --approaches small-zs small-lora laya-zs laya-ft bertimbau
cascade:         ## cascata Laya -> Claude
	$(M).evaluation.cascade --cheap laya-ft --fallback claude
NBRUN = cd notebooks && ../$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=review-intel
notebook-tests:  ## executa notebooks/01_testes_dos_modelos.ipynb (baixa os modelos do HF se preciso)
	$(NBRUN) 01_testes_dos_modelos.ipynb
report:          ## executa notebooks/02_resultados.ipynb e regera docs/figures/
	$(NBRUN) 02_resultados.ipynb
publish-dry:     ## lista o que seria enviado ao Hugging Face
	$(M).hub.publish --dry-run
publish:         ## publica modelos e dataset no Hugging Face (HF_TOKEN no .env)
	$(M).hub.publish

# Pipeline inteiro com rótulos SINTÉTICOS e modelos pequenos: só verifica o encanamento, os números não valem nada.
smoke:
	$(M).labeling.smoke_labels
	$(M).training.qlora --backend mps --model $(SMOKE_MODEL) --labels data/labels/smoke.jsonl --limit 60 --max-steps 10 --warmup 2 --batch-size 4 --grad-accum 1 --output outputs/smoke-lora
	$(M).training.laya --labels data/labels/smoke.jsonl --limit 80 --epochs 1 --grad-accum 1 --output outputs/smoke-laya
	$(M).training.bertimbau --labels data/labels/smoke.jsonl --limit 300 --epochs 1 --output outputs/smoke-bertimbau
	$(M).evaluation.run --labels data/labels/smoke.jsonl --limit 40 --small-model $(SMOKE_MODEL) --lora outputs/smoke-lora --laya-ft outputs/smoke-laya --bertimbau outputs/smoke-bertimbau --out outputs/smoke-eval
	$(M).evaluation.cascade --cheap laya-ft --fallback small-lora --labels data/labels/smoke.jsonl --pred-dir outputs/smoke-eval/predictions
