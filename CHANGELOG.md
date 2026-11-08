# Changelog

All notable architectural, algorithmic, and operational changes to the **Qwen2.5-3B QLoRA & Confidence-Gated Local RAG** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-11-08 (Production Architecture, Confidence-Gated RAG & Modernization Release)

### Highlights
Version `2.0.0` transforms the original consumer-GPU experimental scripts into a production-hardened, reproducibly verified **Hybrid Parametric (QLoRA) + Non-Parametric (Confidence-Gated RAG)** inference and training platform. It eliminates **7 critical legacy codebase defects**, introduces a **FastAPI REST microservice**, adds **deterministic 4-way ablation evidence & SVG telemetry**, and establishes a comprehensive **automated test suite**.

### Added
- **Confidence-Gated RAG Engine ([src/rag_utils.py](src/rag_utils.py))**:
  - Added cosine similarity thresholding (`min_score: float = 0.35`, configurable per call) leveraging L2-normalized `all-MiniLM-L6-v2` embeddings and `FAISS IndexFlatIP` inner-product scores (`D`).
  - Added explicit filtering of FAISS `-1` sentinel indices (returned when `top_k` exceeds the number of indexed passages in `kb/facts.jsonl`) and title-level passage deduplication.
  - Added structured metadata fields (`id`, `score`, `rank`, `title`, `text`) and inline citation formatting (`[1] Title (score=0.75): ...`) in `context_block()`.
- **Word-Boundary Conversational Intent Router ([src/rag_utils.py](src/rag_utils.py), [portable_infer/gradio_app.py](portable_infer/gradio_app.py))**:
  - Implemented `is_conversational_query()` using compiled regex word boundaries (`\b`) and token-length guards so greetings (`"hi"`, `"hello"`, `"thanks"`) cleanly bypass vector search without false-positive collisions on words containing `"hi"` (`"which"`, `"this"`, `"architecture"`, `"machine"`).
- **Completion-Only Assistant Loss Masking ([src/train_qlora_min.py](src/train_qlora_min.py))**:
  - Implemented custom ChatML assistant-turn boundary detection (`<|im_start|>assistant\n`) that masks system and user prompt tokens with `-100` (`--completion_only_loss`, enabled by default), preventing gradient capacity from being wasted on memorizing user questions during Stage 2 micro-patching.
- **FastAPI Production Microservice ([src/api.py](src/api.py))**:
  - Added REST API server with Pydantic request/response schemas, lazy LLM singleton loading, and three endpoints: `GET /health`, `POST /retrieve` (CPU-only `< 8 ms`), and `POST /generate` (supporting dynamic `use_rag`, `use_adapter`, `top_k`, and `min_score` toggles).
- **Deterministic Telemetry & 4-Way Ablation Suite (`src/generate_evidence.py`, `evidence/`)**:
  - Extracted authoritative 150-step training telemetry (`evidence/training_telemetry.json`) and 30-sample evaluation metrics (`evidence/ablation_4way_metrics.json`, `evidence/ablation_4way_summary.md`) from `release_qwen25_qlora.zip`.
  - Generated 3 publication-grade SVG vector charts:
    - [evidence/plots/training_convergence.svg](evidence/plots/training_convergence.svg)
    - [evidence/plots/ablation_4way_comparison.svg](evidence/plots/ablation_4way_comparison.svg)
    - [evidence/plots/vram_budget_4gb.svg](evidence/plots/vram_budget_4gb.svg)
- **Automated Regression Test Suite (`tests/`) & Build Tooling**:
  - Added `unittest` suite covering RAG confidence gating, `-1` sentinel handling, conversational regex routing, completion-only token masking, release archive telemetry parity, and FastAPI endpoints.
  - Added `Makefile`, `Dockerfile`, `.dockerignore`, and `CHECKSUMS.txt`.
- **Hiring-Grade Documentation Architecture**:
  - Authored [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_CARD.md](MODEL_CARD.md), [RUNBOOK.md](RUNBOOK.md), [CHANGELOG.md](CHANGELOG.md), and [CHECKSUMS.txt](CHECKSUMS.txt).

### Fixed (The 7 Legacy v1.0 Defects Resolved)
1. **Fixed Substring Greeting Collision Bug ([portable_infer/gradio_app.py](portable_infer/gradio_app.py))**: Replaced `any(greeting in user.lower() ...)` (which matched `"hi"` inside `"which"`, `"this"`, `"while"`, silently disabling RAG on factual questions) with regex word-boundary intent detection.
2. **Fixed Unconditional RAG Noise Injection & `-1` Sentinel Bug ([src/rag_utils.py](src/rag_utils.py), [src/rag_infer.py](src/rag_infer.py))**: `D, I = _index.search(...)` previously discarded the similarity matrix `D` and indexed `_meta["titles"][idx]` without checking `idx < 0` or `score >= min_score`.
3. **Fixed Dual-Model VRAM Doubling in Evaluation ([src/eval_compare.py](src/eval_compare.py))**: Replaced simultaneous loading of two 4-bit 3B models (`load_base()` + `load_adapter()`, which consumed `~3.8 GB+` weight VRAM and risked OOM on 4 GB GPUs) with a single `PeftModel` instance using `with model.disable_adapter():` (`~2.48 GB` peak VRAM, `50%` faster startup).
4. **Fixed 4x Redundant Model Reload Inside Inference Loop ([src/adapter_infer.py](src/adapter_infer.py))**: Hoisted `load_adapter()` outside `chat(prompt)` into a cached singleton so sequential benchmark prompts do not reload 3B weights from disk on every query.
5. **Fixed Ignored `--max_seq_len` & `--packing` CLI Flags ([src/train_qlora.py](src/train_qlora.py))**: Wired `max_seq_length=args.max_seq_len` and `packing=args.packing` into `SFTTrainer` / `SFTConfig` (previously parsed by `argparse` but never passed to the trainer).
6. **Fixed Full-Sequence Prompt Loss Dilution ([src/train_qlora_min.py](src/train_qlora_min.py))**: Replaced naive `DataCollatorForLanguageModeling(mlm=False)` (which computed loss on boilerplate system/user tokens) with assistant-only completion masking.
7. **Fixed Working-Directory Relative Path Fragility (`src/*.py`, `portable_infer/*.py`, `kb/add_fact.py`)**: Anchored all default paths (`kb/`, `checkpoints/`, `data/processed/`, `outputs/`) to repository root via `pathlib.Path(__file__).resolve()`, eliminating `FileNotFoundError` when launching scripts from subdirectories.

---

## [1.0.0] — 2025-08-27 (Initial Consumer-GPU QLoRA + Local RAG Release)

### Added
- **Stage 1 QLoRA Pipeline**: `src/data_prep.py` (`databricks-dolly-15k` ChatML formatting) and `src/train_qlora.py` (4-bit NF4 double quantization, `r=8`, `lora_alpha=16`, `paged_adamw_8bit`, gradient checkpointing).
- **Trained Checkpoint Artifact**: `qwen25-3b-dolly-qlora-steps150` (`checkpoint-150`, 150 steps, `868,411` tokens, packaged in `release_qwen25_qlora.zip`).
- **Stage 2 Continual Patching Scripts**: `src/make_patch_data.py`, `src/mix_patch.py`, and `src/train_qlora_min.py` for replay-mixed micro-patching.
- **Stage 3 Local FAISS RAG Prototype**: `kb/facts.jsonl`, `src/rag_build.py`, `src/rag_utils.py`, and `src/rag_infer.py` using CPU `all-MiniLM-L6-v2` embeddings and `FAISS IndexFlatIP`.
- **Evaluation & Portable Serving**: `src/eval_compare.py`, `src/summarize_eval.py`, `portable_infer/run.py` (CLI), `portable_infer/gradio_app.py` (Gradio UI), and demo recording (`assets/qwen-local-rag-demo.mp4`).
