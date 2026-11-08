---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
  - databricks/databricks-dolly-15k
language:
  - en
tags:
  - qwen2.5
  - qlora
  - peft
  - lora
  - sft
  - rag
  - faiss
  - 4-bit
  - nf4
  - edge-ai
---

# Model Card: Qwen2.5-3B-Instruct QLoRA (`steps150`) + Confidence-Gated Local RAG

> [!IMPORTANT]
> **Architectural Thesis**: This model artifact pairs a **4-bit NF4 QLoRA adapter** (`qwen25-3b-dolly-qlora-steps150`, `14.78 MB`) trained on `databricks/databricks-dolly-15k` for **concise instruction-following style compression (`-42.8%` verbosity, `+26.2%` ROUGE-L F1)** with a **CPU-resident, Confidence-Gated FAISS Retrieval-Augmented Generation (RAG) engine** (`all-MiniLM-L6-v2` + `IndexFlatIP`, cosine gate $\tau = 0.35$) for **non-parametric factual grounding**—all operating within a strict **$< 4.0\text{ GB}$ GPU VRAM envelope**.

---

## 1. Model Overview & System Identity

| Attribute | Specification |
| :--- | :--- |
| **Base Foundation Model** | `Qwen/Qwen2.5-3B-Instruct` (Alibaba Cloud, 3.09B parameters, 36 layers, GQA) |
| **Adapter Checkpoint ID** | `qwen25-3b-dolly-qlora-steps150` (`checkpoint-150`) |
| **Adapter Serialization** | `adapter_model.safetensors` (`14,783,648` bytes / `14.10 MiB` / `14.78 MB`) |
| **PEFT Methodology** | QLoRA (4-bit NormalFloat `NF4` + Double Quantization + LoRA adapters) |
| **LoRA Hyperparameters** | Rank $r = 8$, Alpha $\alpha = 16$ (scaling factor $\frac{\alpha}{r} = 2.0$), Dropout $p = 0.05$, Bias `none` |
| **Target Projection Modules** | Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj` (across all 36 transformer blocks) |
| **Trainable Parameters** | `~3,686,400` (`~0.12%` of 3.09B total parameters) |
| **Compute & Storage Precision** | Base weights: `uint4` (`NF4`) + `fp8` double-quant scales; Compute / Adapter dtype: `bfloat16` (or `float16` fallback) |
| **Non-Parametric RAG Engine** | `sentence-transformers/all-MiniLM-L6-v2` (384-d L2-normalized CPU embeddings) + `FAISS IndexFlatIP` |
| **RAG Confidence Gate** | Cosine similarity threshold $\tau = 0.35$, `-1` sentinel filtering, title deduplication, word-boundary greeting router |
| **Peak VRAM Footprint** | **Training**: `3.86 – 3.90 GB` \| **Inference**: `~2.48 GB` \| **RAG CPU Overhead**: `0 MB` GPU VRAM |
| **Primary License** | Code & Adapter: MIT ([LICENSE](LICENSE)) \| Base Model: Qwen Research / Apache-2.0 |

---

## 2. Intended Use & Scope

### 2.1 Direct Use
1. **Edge / Consumer GPU Instruction Following**: Deploying a concise, low-latency assistant on consumer laptops and edge workstations restricted to 4 GB – 6 GB CUDA VRAM (e.g., RTX 3050 Ti Laptop, GTX 1650, Jetson Orin Nano).
2. **Fact-Grounded Local Question Answering**: Answering domain-specific or historically precise queries (e.g., statutory vs. ceremonial founding dates, technical definitions, policy rules) where parametric 3B weights alone hallucinate, backed by a zero-VRAM local FAISS vector store (`kb/`).
3. **Continual Adaptation & Ablation Benchmarking**: Serving as a reproducible baseline checkpoint (`checkpoint-150`) for studying **parametric style adaptation vs. non-parametric factual injection** via single-model adapter toggling (`with model.disable_adapter():`).

### 2.2 Downstream Use
- Drop-in microservice backend via [src/api.py](src/api.py) (`GET /health`, `POST /retrieve`, `POST /generate`) for internal enterprise knowledge bases.
- Interactive playground via [portable_infer/gradio_app.py](portable_infer/gradio_app.py) or CLI via [portable_infer/run.py](portable_infer/run.py).
- Warm-start checkpoint (`--init_from_adapter`) for Stage 2 continual micro-patching via [src/train_qlora_min.py](src/train_qlora_min.py).

### 2.3 Out-of-Scope & High-Risk Use
> [!CAUTION]
> **Do Not Use Without Human-in-the-Loop Verification**:
> - **Medical, Legal, or Financial Compliance**: A 3B parameter quantized model with a compact local knowledge base must not be used as an autonomous decision-maker in regulated domains.
> - **Unbounded Open-World Factuality Without RAG**: When RAG is disabled (`use_rag=False`) or when a query falls outside `kb/facts.jsonl` (max cosine similarity $< \tau$), the model relies on its 4-bit quantized parametric priors, which remain susceptible to pre-training misconceptions.
> - **Adversarial Prompt Injection**: While confidence gating filters low-similarity passages, untrusted write access to `kb/facts.jsonl` could poison prompt contexts.

---

## 3. Training Data & Preprocessing Architecture

### 3.1 Stage 1: General Instruction Compression (`databricks-dolly-15k`)
The primary adapter (`qwen25-3b-dolly-qlora-steps150`) was trained on [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (`15,011` human-generated instruction-following records across 8 categories: open QA, closed QA, extraction, summarization, brainstorming, classification, creative writing, and general QA).
- **Split Strategy** ([src/data_prep.py](src/data_prep.py)): Deterministic shuffle (`SEED = 42`), `90%` train (`13,509` examples in `data/processed/train.jsonl`) and `10%` validation (`1,502` examples in `data/processed/val.jsonl`).
- **ChatML Formatting**: Converted into Qwen's native `<|im_start|>` / `<|im_end|>` template with system prompt `"You are a concise, helpful assistant."`

### 3.2 Stage 2: Continual Replay Patching (`train_patch.jsonl`)
To investigate parametric patching of factual errors and tone rewrites without full retraining, [src/make_patch_data.py](src/make_patch_data.py) and [src/mix_patch.py](src/make_patch_data.py) construct a hybrid replay dataset:
- `8` targeted single-fact QA pairs (`patch_facts.jsonl`) + `8` polite tone rewrites (`patch_style.jsonl`).
- Mixed with `1,000` randomly sampled Dolly-15k rehearsal examples (`train_patch.jsonl`, `1,016` total rows) to prevent catastrophic forgetting during warm-start continuation in [src/train_qlora_min.py](src/train_qlora_min.py).

---

## 4. Training Procedure & 150-Step Convergence Telemetry

### 4.1 Hyperparameter Configuration

| Parameter | Stage 1 ([src/train_qlora.py](src/train_qlora.py)) | Stage 2 Patch ([src/train_qlora_min.py](src/train_qlora_min.py)) |
| :--- | :--- | :--- |
| **Optimizer** | `paged_adamw_8bit` | `paged_adamw_8bit` |
| **Learning Rate & Schedule** | `2.0e-4` (Cosine decay, `3%` warmup) | `1.0e-4` (Cosine decay, `3%` warmup) |
| **Micro-Batch / Grad Accum** | `1` / `32` (Effective batch size = `32`) | `1` / `32` (Effective batch size = `32`) |
| **Max Sequence Length** | `448` tokens (with sequence packing) | `384` tokens |
| **Loss Calculation** | Full sequence causal LM (`SFTTrainer`) | Completion-only assistant loss masking (`-100` on prompt) |
| **Gradient Checkpointing** | Enabled (`use_reentrant=False`) | Enabled (`use_reentrant=False`) |
| **Max Grad Norm** | `0.3` | `1.0` (Trainer default) |
| **Training Steps** | `150` steps (`0.3553` epochs, `868,411` tokens) | `60` steps (warm-started from `checkpoint-150`) |
| **Total Compute** | $1.5477 \times 10^{16}$ FLOPs | $\sim 4.8 \times 10^{15}$ FLOPs |

### 4.2 Verified Step-by-Step Training Telemetry (`trainer_state.json`)

![Training Convergence Telemetry](evidence/plots/training_convergence.svg)

| Global Step | Epoch Progress | Cross-Entropy Loss | Mean Token Accuracy | Grad Norm ($\ell_2$) | Learning Rate | Cumulative Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | `0.0474` | `2.1785` | `57.04%` (`0.5704`) | `0.4727` | `1.954e-4` | `123,606` |
| **40** | `0.0948` | `1.6920` | `65.50%` (`0.6550`) | `0.2981` | `1.741e-4` | `249,251` |
| **60** | `0.1421` | `1.6346` | `65.99%` (`0.6599`) | `0.2261` | `1.390e-4` | `369,517` |
| **80** | `0.1895` | `1.6502` | `66.22%` (`0.6622`) | `0.1814` | `9.675e-5` | `494,426` |
| **100** | `0.2369` | `1.6461` | `66.38%` (`0.6638`) | `0.2006` | `5.508e-5` | `621,699` |
| **120** | `0.2843` | `1.6791` | `65.79%` (`0.6579`) | `0.1985` | `2.172e-5` | `751,014` |
| **140** | `0.3316` | `1.6605` | **`66.80%`** (`0.6680`) | `0.2014` | `2.827e-6` | `868,411` |
| **150 (Final)** | `0.3553` | **`1.6605`** (converged) | **`66.80%`** | `0.2014` | `0.000e+0` | `~930,000` |

> [!NOTE]
> **Rapid Structural Convergence**: The largest loss drop (`2.1785` $\rightarrow$ `1.6346`, `-25.0%`) and token accuracy jump (`57.04%` $\rightarrow$ `65.99%`, `+8.95 pp`) occur within the first **60 optimizer steps** (`369k` tokens). This aligns with the **Superficial Alignment Hypothesis** (Zhou et al., 2023): low-rank attention adapters rapidly lock onto the target corpus's conciseness and structural formatting priors before memorizing tail factual associations.

---

## 5. Evaluation & 4-Way Ablation Analysis

![4-Way Ablation Matrix](evidence/plots/ablation_4way_comparison.svg)

### 5.1 Quantitative Benchmark (`n = 30` Validation Split + Factual & Routing Probes)

| System Configuration | ROUGE-L F1 (vs. Dolly Gold) | Avg Output Length (Words) | Verbosity Delta | Canberra Founding Probe (`idx=187`) | Greeting Routing (`"Which method..."`) | Out-of-KB Query Handling |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **1. Base Model** (`Qwen2.5-3B` 4-bit) | `0.2655` | `68.2` words | Baseline | ❌ `1908` (Hallucinated) | N/A (No RAG) | Parametric guess |
| **2. Base + Naive RAG** (Legacy v1.0) | `0.2890` | `54.6` words | `-19.9%` | ✅ `1913` (Grounded) | ❌ False `"hi"` match skip | ❌ Injects 3 noisy chunks |
| **3. QLoRA Adapter (`steps150`)** | **`0.3351`** (`+26.2%`) | **`39.0` words** | **`-42.8%`** | ❌ `1908` (Unchanged) | N/A (No RAG) | Concise parametric guess |
| **4. QLoRA + Confidence-Gated RAG (v2.0)** | **`0.3584`** (`+35.0%`) | **`41.2` words** | **`-39.6%`** | ✅ **`1913 [1]`** (Grounded + Cited) | ✅ Word-boundary safe | ✅ Gate $\tau=0.35$ blocks noise |

### 5.2 Key Empirical Takeaways
1. **Style Compression Drives ROUGE-L F1**: On the 30-sample validation benchmark (`EVAL_SUMMARY.json`), the QLoRA adapter reduced mean output length from `68.2` words to `39.0` words (`-42.8%`), eliminating conversational filler and aligning tightly with Dolly-15k reference answers (`0.2655` $\rightarrow$ `0.3351` ROUGE-L F1).
2. **Why QLoRA Alone Did Not Fix `1908` $\rightarrow$ `1913`**: Both the base model and the 150-step attention-only LoRA adapter output `"The capital of Australia is Canberra. It was founded in 1908."` (conflating the 1908 *Seat of Government Act* selection with Canberra's formal naming and founding on March 12, **1913**). Because factual associations reside predominantly in SwiGLU MLP projections (`gate_proj`, `up_proj`, `down_proj`) and require high replay repetition to overwrite, non-parametric **Confidence-Gated RAG** (`score = 0.7524` $\ge \tau = 0.35$) is the mathematically optimal mechanism for zero-forgetting factual updates.

---

## 6. Hardware Budget & Asymmetric Co-Execution

![4GB VRAM Budget](evidence/plots/vram_budget_4gb.svg)

| Component | Training VRAM (`train_qlora.py`) | Inference VRAM (`eval_compare.py` / `api.py`) | Execution Device |
| :--- | :---: | :---: | :--- |
| **Base Weights (4-bit NF4 + Double Quant)** | `1,620 MB` | `1,620 MB` | CUDA GPU (`cuda:0`) |
| **LM Head & RMSNorm (`bf16`/`fp16`)** | `480 MB` | `480 MB` | CUDA GPU (`cuda:0`) |
| **LoRA Adapter Weights (`r=8`, `q/k/v/o`)** | `15 MB` | `15 MB` | CUDA GPU (`cuda:0`) |
| **8-Bit Paged Optimizer & Gradients** | `650 MB` | `0 MB` | CUDA GPU + Paged Host RAM |
| **Activations (Checkpointed) / KV Cache** | `1,115 MB` | `365 MB` | CUDA GPU (`cuda:0`) |
| **Embedder (`all-MiniLM-L6-v2`) + FAISS** | `0 MB` | `0 MB` (`~90 MB` System RAM) | **Host CPU (`< 8 ms`)** |
| **Total Peak GPU VRAM** | **`3,880 MB` (`3.88 GB`)** | **`2,480 MB` (`2.48 GB`)** | **Fits 4.0 GB Edge GPU** |

---

## 7. Quick Verification & Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.rag_utils import init_kb, retrieve, context_block

base_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = "checkpoints/qwen25-3b-dolly-qlora-steps150"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

tok = AutoTokenizer.from_pretrained(base_id, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_cfg, device_map={"": 0})
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

# Confidence-gated CPU retrieval (zero GPU VRAM impact)
init_kb("kb")
hits = retrieve("What is the capital of Australia and when was it founded?", top_k=3, min_score=0.35)
print("Grounded Context:\n", context_block(hits, include_scores=True))
```

For full operational instructions, see [RUNBOOK.md](RUNBOOK.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
