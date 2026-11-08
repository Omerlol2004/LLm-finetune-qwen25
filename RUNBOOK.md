# Operational Runbook: Qwen2.5-3B QLoRA & Confidence-Gated Local RAG

> [!IMPORTANT]
> **Scope of This Runbook**: Standard Operating Procedures (SOPs) for environment provisioning, deterministic artifact verification, Stage 1 & Stage 2 QLoRA training under a `< 4.0 GB` VRAM ceiling, FAISS knowledge-base indexing, multi-surface inference (CLI, Gradio UI, FastAPI microservice), 4-way evaluation, and incident troubleshooting.

---

## Table of Contents

1. [System Prerequisites & Hardware Verification](#1-system-prerequisites--hardware-verification)
2. [Environment Provisioning & Release Unpacking](#2-environment-provisioning--release-unpacking)
3. [Offline Verification & Regression Test Suite](#3-offline-verification--regression-test-suite)
4. [SOP-01: Data Preparation & Replay Patch Mixing](#4-sop-01-data-preparation--replay-patch-mixing)
5. [SOP-02: QLoRA Training (Stage 1 SFT & Stage 2 Patching)](#5-sop-02-qlora-training-stage-1-sft--stage-2-patching)
6. [SOP-03: FAISS Knowledge Base Management](#6-sop-03-faiss-knowledge-base-management)
7. [SOP-04: Inference & Production Serving (CLI, Gradio, FastAPI)](#7-sop-04-inference--production-serving-cli-gradio-fastapi)
8. [SOP-05: 4-Way Evaluation & Telemetry Regeneration](#8-sop-05-4-way-evaluation--telemetry-regeneration)
9. [Incident Response & Troubleshooting Playbook](#9-incident-response--troubleshooting-playbook)

---

## 1. System Prerequisites & Hardware Verification

### 1.1 Hardware Tier Matrix

| Tier | GPU VRAM | System RAM | Supported Workflows |
| :--- | :--- | :--- | :--- |
| **Tier 0: CI / CPU-Only** | `0 GB` (No GPU) | `4 GB+` | Unit tests (`tests/`), FAISS KB build & retrieval (`< 8 ms`), FastAPI `/health` & `/retrieve`, telemetry plot generation |
| **Tier 1: Edge CUDA (Target)** | **`4.0 GB VRAM`** | `8 GB – 16 GB` | Full 4-bit NF4 Inference (`~2.48 GB`), Single-Model Base-vs-Adapter Eval, and 4-bit QLoRA Training (`3.86 – 3.90 GB`) |
| **Tier 2: Workstation CUDA** | `8.0 GB+ VRAM` | `16 GB+` | Longer context lengths (`max_seq_len >= 1024`), larger micro-batches, and `all-linear` LoRA/DoRA expansion |

### 1.2 Verify CUDA & Device Capability
Run the hardware diagnostic probe before launching GPU workloads:

```bash
python3 src/verify_gpu.py
```

Expected healthy output on a CUDA workstation/laptop:
```text
PyTorch: 2.4.0+cu121
CUDA available: True
Device count: 1
Device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
bf16 supported: True
```

---

## 2. Environment Provisioning & Release Unpacking

### 2.1 Virtual Environment & Dependencies

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Extracting the Pre-Trained Adapter Checkpoint (`checkpoint-150`)
The repository includes `release_qwen25_qlora.zip` containing the trained `qwen25-3b-dolly-qlora-steps150` adapter weights (`adapter_model.safetensors`), tokenizer artifacts, TensorBoard event logs, and `EVAL_SUMMARY.json`.

To unpack the trained checkpoint into `checkpoints/qwen25-3b-dolly-qlora-steps150` for immediate inference:

```bash
python3 -c '
import zipfile, pathlib
 dst = pathlib.Path("checkpoints")
 dst.mkdir(exist_ok=True)
 with zipfile.ZipFile("release_qwen25_qlora.zip") as z:
     for m in z.namelist():
         norm = m.replace("\\", "/")
         if norm.startswith("qwen25-3b-dolly-qlora-steps150/"):
             out_path = dst / norm
             if m.endswith("/") or m.endswith("\\"):
                 out_path.mkdir(parents=True, exist_ok=True)
             else:
                 out_path.parent.mkdir(parents=True, exist_ok=True)
                 out_path.write_bytes(z.read(m))
print("Extracted to checkpoints/qwen25-3b-dolly-qlora-steps150")
'
```

---

## 3. Offline Verification & Regression Test Suite

Before deploying or modifying any component, run the deterministic verification suite (executes in `< 15 seconds` on CPU with zero network/GPU dependency):

```bash
# Run the complete unit and integration test suite
python3 -m unittest discover -s tests -v

# Verify SHA-256 artifact integrity
sha256sum -c CHECKSUMS.txt
```

What the automated test suite validates:
1. **Confidence-Gated Retrieval (`src/rag_utils.py`)**: Cosine threshold filtering (`min_score`), `-1` FAISS empty-slot protection, duplicate title deduplication, and inline `[rank] Title (score=...)` formatting.
2. **Word-Boundary Conversational Router (`src/rag_utils.py` & `gradio_app.py`)**: Ensures greetings (`"hi"`, `"hello"`, `"thanks"`) bypass RAG while factual queries containing `"hi"` as a substring (`"Which study techniques..."`, `"What is the architecture of this model?"`) are **never** falsely skipped.
3. **Completion-Only Assistant Loss Masking (`src/train_qlora_min.py`)**: Verifies that system/user tokens receive label `-100` and only assistant response tokens contribute to cross-entropy gradients.
4. **Release Archive & Telemetry Integrity (`evidence/`)**: Confirms exact parity between `release_qwen25_qlora.zip`, `evidence/training_telemetry.json`, `evidence/ablation_4way_metrics.json`, and the 3 SVG plots.
5. **FastAPI Service Contracts (`src/api.py`)**: Tests `/health` and `/retrieve` endpoints and schema validation.

---

## 4. SOP-01: Data Preparation & Replay Patch Mixing

### 4.1 Stage 1 Dataset Preparation (`databricks-dolly-15k`)
Downloads `databricks/databricks-dolly-15k`, formats each entry into Qwen ChatML messages (`system`, `user`, `assistant`), and writes a deterministic 90/10 train/validation split:

```bash
python3 src/data_prep.py
```
- **Outputs**: `data/processed/train.jsonl` (`13,509` rows) and `data/processed/val.jsonl` (`1,502` rows).

### 4.2 Stage 2 Continual Patch Dataset Construction
Generates curated factual QA and tone-rewrite patches and mixes them with `1,000` rehearsal samples from `train.jsonl` to mitigate catastrophic forgetting:

```bash
# Step 1: Generate data/processed/patch_facts.jsonl (8 rows) & patch_style.jsonl (8 rows)
python3 src/make_patch_data.py

# Step 2: Mix patches with 1,000 Dolly-15k replay rows -> data/processed/train_patch.jsonl (1,016 rows)
python3 src/mix_patch.py
```

---

## 5. SOP-02: QLoRA Training (Stage 1 SFT & Stage 2 Patching)

### 5.1 Stage 1: Primary QLoRA Fine-Tuning (`src/train_qlora.py`)

```bash
python3 src/train_qlora.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --train_jsonl data/processed/train.jsonl \
  --val_jsonl data/processed/val.jsonl \
  --output_dir checkpoints/qwen25-3b-dolly-qlora-steps150 \
  --max_seq_len 448 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-4 \
  --max_steps 150 \
  --logging_steps 20 \
  --packing
```

### 5.2 Stage 2: Memory-Conscious Continual Patching (`src/train_qlora_min.py`)
Warm-starts from `checkpoint-150` using **completion-only assistant loss masking** (system and user prompt tokens masked with `-100`):

```bash
python3 src/train_qlora_min.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --train_jsonl data/processed/train_patch.jsonl \
  --val_jsonl data/processed/val.jsonl \
  --init_from_adapter checkpoints/qwen25-3b-dolly-qlora-steps150 \
  --output_dir checkpoints/qwen25-3b-dolly-qlora-steps210_patch \
  --max_steps 60 \
  --max_seq_len 384 \
  --learning_rate 1e-4
```

---

## 6. SOP-03: FAISS Knowledge Base Management

### 6.1 Adding or Updating Facts
Facts are stored as line-delimited JSON objects in [kb/facts.jsonl](kb/facts.jsonl):
```json
{"id": "au_canberra_1913", "title": "Canberra founding", "text": "The capital of Australia is Canberra. The name 'Canberra' was officially adopted and the city founded in 1913, following a decision to create a new capital rather than choose Sydney or Melbourne."}
```
You can append a new fact interactively via `python3 kb/add_fact.py` or by editing `kb/facts.jsonl` directly.

### 6.2 Rebuilding the CPU FAISS Index
Whenever `kb/facts.jsonl` is modified, rebuild `kb/index.faiss` and `kb/meta.json`:

```bash
python3 src/rag_build.py
```
- Uses `sentence-transformers/all-MiniLM-L6-v2` on CPU (`device="cpu"`).
- Normalizes embeddings to unit $\ell_2$ norm (`normalize_embeddings=True`) and indexes them into `faiss.IndexFlatIP(384)`, making inner product identical to exact cosine similarity.

---

## 7. SOP-04: Inference & Production Serving (CLI, Gradio, FastAPI)

### 7.1 Interactive CLI (`portable_infer/run.py`)

```bash
# Run with QLoRA + Confidence-Gated RAG (recommended)
python3 portable_infer/run.py --rag --top_k 3 --min_score 0.35

# Run with QLoRA adapter only (no RAG)
python3 portable_infer/run.py
```

### 7.2 Interactive Gradio Web UI (`portable_infer/gradio_app.py`)

```bash
python3 portable_infer/gradio_app.py
```
Open `http://localhost:7860` in your browser. Toggle **Use RAG (local KB)** and adjust **Top-K** (`1–5`) and **Min Confidence Score** (`0.0–0.8`, default `0.35`).

### 7.3 FastAPI Production Microservice (`src/api.py`)

Launch the REST server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### Endpoint 1: Health & Readiness Check (`GET /health`)
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
```json
{
  "status": "ok",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "adapter_dir": "checkpoints/qwen25-3b-dolly-qlora-steps150",
  "kb_ready": true,
  "kb_passages": 3,
  "model_loaded": false,
  "cuda_available": true
}
```

#### Endpoint 2: Confidence-Gated Retrieval Only (`POST /retrieve`)
Executes in `< 8 ms` on CPU without loading the 3B LLM into GPU VRAM:
```bash
curl -s -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of Australia and when was it founded?", "top_k": 3, "min_score": 0.35}' \
  | python3 -m json.tool
```
```json
{
  "query": "What is the capital of Australia and when was it founded?",
  "is_conversational": false,
  "hits": [
    {
      "rank": 1,
      "id": "au_canberra_1913",
      "title": "Canberra founding",
      "text": "The capital of Australia is Canberra. The name 'Canberra' was officially adopted and the city founded in 1913, following a decision to create a new capital rather than choose Sydney or Melbourne.",
      "score": 0.7524
    }
  ],
  "context_formatted": "[1] Canberra founding (score=0.75): The capital of Australia is Canberra. The name 'Canberra' was officially adopted and the city founded in 1913, following a decision to create a new capital rather than choose Sydney or Melbourne.",
  "latency_ms": 5.42
}
```

#### Endpoint 3: End-to-End RAG + QLoRA Generation (`POST /generate`)
```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of Australia and when was it founded?",
    "use_rag": true,
    "use_adapter": true,
    "top_k": 3,
    "min_score": 0.35,
    "max_new_tokens": 128,
    "temperature": 0.2
  }' | python3 -m json.tool
```

---

## 8. SOP-05: 4-Way Evaluation & Telemetry Regeneration

### 8.1 Single-Model In-Place Base vs. Adapter Evaluation (`src/eval_compare.py`)
In v2.0, `src/eval_compare.py` loads the 4-bit base model **once**, wraps it with `PeftModel`, and uses `with model.disable_adapter():` to evaluate the base model vs. the LoRA adapter in-place—saving `~1.75 GB` VRAM and preventing OOM crashes on 4 GB GPUs:

```bash
python3 src/eval_compare.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter checkpoints/qwen25-3b-dolly-qlora-steps150 \
  --val_path data/processed/val.jsonl \
  --num_samples 30 \
  --max_new_tokens 128

python3 src/summarize_eval.py
```

### 8.2 Regenerating Evidence Tables & SVG Telemetry Plots
```bash
python3 src/generate_evidence.py
```
Regenerates:
- `evidence/training_telemetry.json`
- `evidence/ablation_4way_metrics.json` & `evidence/ablation_4way_summary.md`
- `evidence/plots/training_convergence.svg`
- `evidence/plots/ablation_4way_comparison.svg`
- `evidence/plots/vram_budget_4gb.svg`

---

## 9. Incident Response & Troubleshooting Playbook

| Symptom / Error Signature | Root Cause | Remediation SOP |
| :--- | :--- | :--- |
| `torch.cuda.OutOfMemoryError: CUDA out of memory` during training | Sequence length or batch size exceeds 4.0 GB VRAM ceiling, or another process holds GPU context. | 1. Ensure `--per_device_train_batch_size 1` and `--gradient_accumulation_steps 32`.<br>2. Lower `--max_seq_len` from `448` to `384` (or use `src/train_qlora_min.py`).<br>3. Verify `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` and `optim="paged_adamw_8bit"` are active. |
| `CUDA out of memory` during `eval_compare.py` | Legacy dual-model loading instantiated two separate 3B models (`~3.6 GB` weights alone). | Use the updated v2.0 `src/eval_compare.py`, which shares a single base model and toggles `with model.disable_adapter():` (`~2.48 GB` peak VRAM). |
| `FileNotFoundError: KB not built. Missing .../index.faiss` | Running scripts from different working directories (`portable_infer/` vs repo root) or missing index build. | 1. Run `python3 src/rag_build.py` to compile `kb/index.faiss`.<br>2. All v2.0 scripts resolve paths relative to `Path(__file__).resolve().parents[1]`, so they work from any `cwd`. |
| RAG skips factual questions like `"Which study technique is best?"` | Legacy v1.0 substring bug (`"hi" in user.lower()` matched `"w**hi**ch"`). | Fixed in v2.0 via `is_conversational_query()` regex word-boundary matching (`\b(?:hi|hello|hey|...)\b`). Verify with `python3 -m unittest discover -s tests -v`. |
| Irrelevant facts injected into non-KB prompts | Legacy v1.0 `retrieve()` ignored FAISS inner-product scores `D` and injected all `top_k` hits unconditionally. | Keep `--min_score 0.35` enabled (`retrieve(..., min_score=0.35)`). Passages with cosine similarity $< 0.35$ are automatically filtered out. |
| `PackageNotFoundError: No package metadata was found for bitsandbytes` on CPU/macOS | `bitsandbytes` requires a CUDA-capable Linux/Windows environment for 4-bit NF4 kernels. | On CPU-only hosts, v2.0 scripts gracefully fall back to unquantized CPU execution or run retrieval-only/verification workflows (`tests/`, `src/rag_build.py`, `/retrieve`). |
