.PHONY: help install test lint build-kb train train-patch eval summarize serve ui cli

PYTHON ?= python3

help:
	@echo "Qwen2.5-3B QLoRA + Confidence-Gated RAG Automation Targets:"
	@echo "  make install       - Install dependencies in editable mode"
	@echo "  make test          - Run full CPU-safe pytest suite"
	@echo "  make lint          - Bytecode-compile all Python modules"
	@echo "  make build-kb      - Rebuild FAISS index (kb/index.faiss, kb/meta.json) from kb/facts.jsonl"
	@echo "  make train         - Run primary QLoRA SFT training (150 steps)"
	@echo "  make train-patch   - Generate & mix upsampled patch dataset and run completion-masked patch training"
	@echo "  make eval          - Run single-model Base vs. QLoRA Adapter (+RAG) evaluation"
	@echo "  make summarize     - Aggregate outputs/eval_compare.jsonl into outputs/EVAL_SUMMARY.json"
	@echo "  make serve         - Launch FastAPI microservice on port 8000"
	@echo "  make ui            - Launch Gradio Web UI on port 7860"
	@echo "  make cli           - Launch interactive Confidence-Gated RAG CLI"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src portable_infer kb test_canberra.py tests

build-kb:
	$(PYTHON) src/rag_build.py

train:
	$(PYTHON) src/train_qlora.py --max_steps 150 --eval_steps 50 --save_steps 50 --output_dir checkpoints/qwen25-3b-dolly-qlora-steps150

train-patch:
	$(PYTHON) src/make_patch_data.py
	$(PYTHON) src/mix_patch.py --patch_repeat 8 --sample_main 250
	$(PYTHON) src/train_qlora_min.py --max_steps 60

eval:
	$(PYTHON) src/eval_compare.py --num_samples 30 --max_new_tokens 128

summarize:
	$(PYTHON) src/summarize_eval.py

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) portable_infer/gradio_app.py

cli:
	$(PYTHON) portable_infer/run.py --rag --score_threshold 0.30
