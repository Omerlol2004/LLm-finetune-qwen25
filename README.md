# QLoRA Fine-Tuning for Qwen2.5-3B with Local RAG

This project implements a compact fine-tuning and inference pipeline for `Qwen/Qwen2.5-3B-Instruct`. It uses QLoRA to adapt the model on instruction-following data, then adds an optional local Retrieval-Augmented Generation (RAG) layer for factual grounding during inference.

The main focus is practical experimentation on consumer hardware: training a 3B parameter model with 4-bit quantization, evaluating the adapter against the base model, and packaging the result into a portable CLI and Gradio web interface.

## Table of Contents

- [Project Objective](#project-objective)
- [Key Results](#key-results)
- [System Overview](#system-overview)
- [Core Workflow](#core-workflow)
- [Training Methodology](#training-methodology)
- [RAG Integration](#rag-integration)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Release Package](#release-package)
- [Hardware Requirements](#hardware-requirements)
- [Current Status and Limitations](#current-status-and-limitations)
- [Future Work](#future-work)
- [License and Attribution](#license-and-attribution)

## Project Objective

The goal is to build an end-to-end LLM adaptation workflow that can:

1. Prepare instruction-following data in Qwen chat format.
2. Fine-tune `Qwen2.5-3B-Instruct` using QLoRA on limited VRAM.
3. Compare base-model and adapter responses with ROUGE-based evaluation.
4. Add a small local FAISS knowledge base for factual correction.
5. Run the fine-tuned model through a command-line interface.
6. Provide a simple Gradio UI for interactive testing.
7. Package the trained adapter, inference code, knowledge base, and docs into a release archive.

## Key Results

The original experiment reported the following metrics for the included QLoRA workflow:

| Metric | Base Model | Fine-Tuned Adapter | Result |
| --- | ---: | ---: | --- |
| ROUGE-L F1 | 0.2655 | 0.3351 | +26.2% |
| Average response length | 127.8 chars | 185.2 chars | +45% |
| Training steps | - | 150 | Short consumer-GPU run |
| Peak VRAM | - | 3.86-3.90 GB | Fits 4 GB GPU class |
| Inference latency | - | ~3.43 s | Measured on project setup |

The project also demonstrates a RAG correction case:

| Query | Without RAG | With RAG | Outcome |
| --- | --- | --- | --- |
| "When was Canberra founded?" | 1908 | 1913 | RAG retrieves the local fact and corrects the answer |

These numbers are useful as project evidence, but they should be treated as experiment-specific results. Re-running training or evaluation on different hardware, seeds, dependency versions, or adapters may produce different values.

## System Overview

```text
Dolly-15k dataset
   |
   v
Chat-format data preparation
   |
   v
Qwen2.5-3B-Instruct base model
   |
   v
4-bit QLoRA fine-tuning
   |
   v
LoRA adapter checkpoint
   |
   +--------------------+
   |                    |
   v                    v
Base vs adapter eval    Portable inference
                        |
                        v
                 Optional local RAG
                        |
                        v
               CLI or Gradio web UI
```

## Core Workflow

### 1. Data Preparation

`src/data_prep.py` loads the Databricks Dolly-15k dataset and converts each example into Qwen-style chat messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise, helpful assistant."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

The script writes:

```text
data/processed/train.jsonl
data/processed/val.jsonl
```

These generated files are ignored by Git.

### 2. QLoRA Training

The main training scripts load `Qwen/Qwen2.5-3B-Instruct` in 4-bit mode and train LoRA adapters on attention projection layers.

Primary scripts:

- `src/train_qlora.py` - main training path using TRL `SFTTrainer`
- `src/train_qlora_min.py` - memory-conscious continuation or patch-training path

### 3. Evaluation

`src/eval_compare.py` compares the base model and the LoRA adapter on validation samples. It generates responses from both models and computes ROUGE scores against the reference answers.

The evaluation output is written to:

```text
outputs/eval_compare.jsonl
```

### 4. RAG Knowledge Base

The project includes a small local knowledge base under `kb/`:

```text
kb/
|-- facts.jsonl
|-- index.faiss
`-- meta.json
```

`src/rag_build.py` rebuilds the FAISS index from `facts.jsonl`, and `src/rag_utils.py` handles retrieval during inference.

### 5. Portable Inference

The `portable_infer/` folder contains a standalone CLI and Gradio app that load the base model plus LoRA adapter, with an optional RAG toggle.

## Training Methodology

| Area | Choice |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank | `r=8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimizer | `paged_adamw_8bit` |
| Sequence length | 448 in the main script, 384 in the memory-optimized script |
| Gradient accumulation | 32 |
| Dataset | `databricks/databricks-dolly-15k` |

The setup is designed to reduce memory pressure:

- The base model is loaded in 4-bit.
- LoRA trains only a small adapter instead of all model weights.
- Gradient checkpointing trades compute for memory.
- Small per-device batch size keeps VRAM usage low.
- Gradient accumulation preserves a larger effective batch.

## RAG Integration

The RAG layer is intentionally lightweight:

1. Facts are stored in `kb/facts.jsonl`.
2. `sentence-transformers/all-MiniLM-L6-v2` embeds each fact on CPU.
3. FAISS stores normalized vectors using inner-product search.
4. At inference time, the query retrieves top-k facts.
5. Retrieved facts are inserted into the model prompt as context.

Example knowledge-base fact:

```json
{
  "id": "au_canberra_1913",
  "title": "Canberra founding",
  "text": "The capital of Australia is Canberra. The name 'Canberra' was officially adopted and the city founded in 1913..."
}
```

The purpose is not to build a large production RAG system. It demonstrates how even a small local knowledge base can correct factual gaps in an adapted model.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- test_canberra.py
|-- release_qwen25_qlora.zip
|-- kb/
|   |-- facts.jsonl
|   |-- index.faiss
|   |-- meta.json
|   `-- add_fact.py
|-- portable_infer/
|   |-- README.md
|   |-- requirements.txt
|   |-- run.py
|   `-- gradio_app.py
|-- scripts/
|   |-- run_cli_norag.bat
|   |-- run_cli_rag.bat
|   |-- run_eval_30.bat
|   `-- make_release.bat
`-- src/
    |-- data_prep.py
    |-- train_qlora.py
    |-- train_qlora_min.py
    |-- eval_compare.py
    |-- summarize_eval.py
    |-- rag_build.py
    |-- rag_infer.py
    |-- rag_utils.py
    |-- quick_infer.py
    |-- adapter_infer.py
    |-- make_patch_data.py
    `-- mix_patch.py
```

Generated artifacts are intentionally ignored by Git:

```text
data/processed/
checkpoints/
outputs/
```

The release zip is included for distribution of the trained adapter and generated outputs.

## Installation

Clone the repository:

```bash
git clone https://github.com/Omerlol2004/LLm-finetune-qwen25.git
cd LLm-finetune-qwen25
```

Create an environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify CUDA/GPU availability:

```bash
python src\verify_gpu.py
```

The dependency file uses the CUDA 12.1 PyTorch index. If your machine uses a different CUDA setup, install the matching PyTorch build first, then install the remaining dependencies.

## Usage

### Run CLI Inference Without RAG

```bat
scripts\run_cli_norag.bat
```

This launches `portable_infer/run.py` with the LoRA adapter and no retrieval context.

### Run CLI Inference With RAG

```bat
scripts\run_cli_rag.bat
```

This enables retrieval from the local `kb/` directory.

### Launch the Gradio Web UI

```bash
cd portable_infer
python gradio_app.py
```

Open:

```text
http://localhost:7860
```

### Rebuild the RAG Index

After editing `kb/facts.jsonl`, rebuild the FAISS index:

```bash
python src\rag_build.py
```

### Add a Knowledge-Base Fact

```bash
cd kb
python add_fact.py
```

## Training From Scratch

Prepare Dolly-15k data:

```bash
python src\data_prep.py
```

Run the main QLoRA training script:

```bash
python src\train_qlora.py
```

Run the memory-optimized continuation script:

```bash
python src\train_qlora_min.py
```

Important default paths:

```text
Base model:      Qwen/Qwen2.5-3B-Instruct
Train data:      data/processed/train.jsonl
Validation data: data/processed/val.jsonl
Adapter output:  checkpoints/qwen25-3b-dolly-qlora
```

## Evaluation

Run the 30-sample evaluation batch file:

```bat
scripts\run_eval_30.bat
```

Or call the evaluator directly:

```bash
python src\eval_compare.py ^
  --adapter checkpoints\qwen25-3b-dolly-qlora-steps150 ^
  --num_samples 30
```

The evaluator:

- Loads the base Qwen model.
- Loads the LoRA adapter.
- Samples validation examples.
- Generates base and adapter answers.
- Computes ROUGE scores.
- Writes detailed rows to `outputs/eval_compare.jsonl`.

## Release Package

Create a portable release archive:

```bat
scripts\make_release.bat
```

The repository currently includes:

```text
release_qwen25_qlora.zip
```

The release package is intended to contain the trained adapter, portable inference tools, knowledge base, evaluation outputs, scripts, and documentation.

## Hardware Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| GPU | 4 GB VRAM CUDA GPU | 6 GB+ VRAM |
| System RAM | 8 GB | 16 GB+ |
| Python | 3.10 recommended | 3.10 recommended |
| Storage | Enough for model downloads and adapter artifacts | SSD recommended |

The base model downloads from Hugging Face at runtime. Make sure your environment has network access and any required model permissions.

## Current Status and Limitations

Current strengths:

- End-to-end data preparation, fine-tuning, evaluation, RAG, and inference workflow.
- Practical QLoRA settings for a 4 GB VRAM training target.
- Base-vs-adapter comparison script with reproducible output files.
- Local RAG demo that corrects a concrete factual query.
- Portable CLI and Gradio interfaces for testing the adapter.
- Release packaging script and included release archive.

Known limitations:

- `checkpoints/`, `outputs/`, and `data/processed/` are ignored, so users need the release archive or must regenerate those artifacts.
- Evaluation is currently ROUGE-based; it does not fully measure factuality, helpfulness, or instruction following.
- The RAG knowledge base is intentionally small and demonstration-focused.
- Most helper scripts are Windows-oriented batch files.
- The Gradio UI loads the model at startup, so first launch may take time while downloading or loading model weights.
- Large model loading depends on local CUDA, PyTorch, BitsAndBytes, and Hugging Face compatibility.

## Future Work

1. Expand evaluation:
   - Add exact-match factual tests.
   - Add human preference or rubric-based evaluation.
   - Track hallucination rate with and without RAG.
   - Compare multiple adapter checkpoints.

2. Improve RAG:
   - Add more domain-specific facts.
   - Add source citations in the final answer.
   - Add relevance thresholds before injecting context.
   - Add reranking for retrieved passages.

3. Improve training:
   - Run longer training on higher-VRAM hardware.
   - Add config files for repeatable experiments.
   - Save training logs and metric summaries in a documented format.
   - Compare LoRA ranks and target module choices.

4. Improve portability:
   - Add Linux/macOS shell scripts.
   - Add Docker setup.
   - Add `.env.example`.
   - Document how to unpack and use the release archive.

## License and Attribution

This project is released under the terms described in [LICENSE](LICENSE).

The project builds on:

- `Qwen/Qwen2.5-3B-Instruct` by Alibaba Cloud
- Databricks Dolly-15k
- QLoRA methodology
- Hugging Face Transformers, PEFT, TRL, Datasets, and Evaluate
- FAISS and Sentence Transformers for local retrieval
