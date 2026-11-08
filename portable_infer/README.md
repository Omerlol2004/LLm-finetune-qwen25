# Portable Inference (CLI & Gradio Web UI)

Self-contained inference runners with root-anchored paths (can be launched from either the repository root or `portable_infer/`) and **Confidence-Gated Hybrid RAG** ($\tau = 0.30$).

## Setup

```bash
pip install -r portable_infer/requirements.txt
```

## Interactive CLI

```bash
# Direct QLoRA adapter inference
python portable_infer/run.py

# Confidence-Gated Hybrid RAG (default threshold tau = 0.30)
python portable_infer/run.py --rag --score_threshold 0.30

# Strict KB mode (abstains when no passage meets tau)
python portable_infer/run.py --rag --score_threshold 0.30 --strict_kb
```

## Gradio Web UI

```bash
python portable_infer/gradio_app.py
```

Open `http://localhost:7860` to interactively adjust `Top-K`, `Score Threshold (τ)`, and `Strict KB Only`.
Requires ~2.6 GB free VRAM in 4-bit NF4 quantization; the base model (`Qwen/Qwen2.5-3B-Instruct`) downloads automatically from Hugging Face Hub on first run.