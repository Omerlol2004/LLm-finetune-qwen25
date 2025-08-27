# QLoRA Fine-tuning: Qwen2.5-3B + RAG Integration 🚀

**Status: ✅ PRODUCTION READY** | **ROUGE-L Improvement: +26.2%** | **4GB VRAM Compatible**

## 🎯 Project Overview

Complete QLoRA fine-tuning pipeline for **Qwen2.5-3B-Instruct** with RAG integration, optimized for consumer hardware. Successfully trained on RTX 3050 4GB with significant performance improvements.

### 📊 Key Results
- **Training**: 150 steps, 3.86-3.90 GB peak VRAM
- **Performance**: ROUGE-L F1 0.2655 → 0.3351 (+26.2%)
- **RAG Integration**: Factual accuracy improvement (Canberra: 1908 → 1913 correct)
- **Inference**: ~3.43s latency for complex queries

### 🛠️ Tech Stack
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Method**: QLoRA (4-bit NF4, LoRA r=8)
- **Dataset**: Dolly-15k (13,509 train / 1,502 val)
- **RAG**: Local FAISS + sentence-transformers
- **Hardware**: RTX 3050 4GB (consumer GPU)

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
python src\verify_gpu.py
```

### 2. Run Inference (CLI)
```bash
# Without RAG
scripts\run_cli_norag.bat

# With RAG (factual grounding)
scripts\run_cli_rag.bat
```

### 3. Launch Web UI
```bash
cd portable_infer
python gradio_app.py
# Open: http://localhost:7860
```

### 4. Run Evaluation
```bash
scripts\run_eval_30.bat
```

## 📁 Project Structure

```
llm-finetune-qwen25/
├── 📊 Training & Evaluation
│   ├── src/
│   │   ├── train_qlora.py          # Main training script
│   │   ├── train_qlora_min.py      # Memory-optimized training
│   │   ├── eval_compare.py         # A/B evaluation (ROUGE-L)
│   │   └── summarize_eval.py       # Generate evaluation summary
│   └── scripts/
│       ├── run_eval_30.bat         # Quick 30-sample evaluation
│       └── make_release.bat        # Package release bundle
│
├── 🤖 Inference & RAG
│   ├── portable_infer/
│   │   ├── run.py                  # CLI inference (RAG toggle)
│   │   ├── gradio_app.py          # Web UI interface
│   │   └── requirements.txt        # Portable dependencies
│   ├── scripts/
│   │   ├── run_cli_norag.bat      # CLI without RAG
│   │   └── run_cli_rag.bat        # CLI with RAG
│   └── kb/                         # Knowledge base
│       ├── facts.jsonl            # Fact database
│       └── index.faiss            # Vector index
│
├── 🎯 Model & Results
│   ├── checkpoints/
│   │   └── qwen25-3b-dolly-qlora-steps150/  # Trained adapter
│   └── outputs/
│       ├── FINAL_REPORT.md        # Technical report
│       ├── EVAL_SUMMARY.json      # Machine-readable metrics
│       ├── eval_compare.jsonl     # Detailed evaluation results
│       └── logs/                  # Training and evaluation logs
│
└── 📋 Documentation
    ├── README.md                   # This file
    ├── LICENSE                     # MIT License with third-party components
    └── release_qwen25_qlora.zip   # Complete release package
```

## 🎯 Training Results

### Performance Metrics
| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| ROUGE-L F1 | 0.2655 | 0.3351 | **+26.2%** |
| Avg Response Length | 127.8 chars | 185.2 chars | +45% |
| Training VRAM | - | 3.86-3.90 GB | ✅ 4GB Compatible |
| Inference Latency | - | 3.43s | ⚡ Fast |

### RAG Factual Correction
| Query | Without RAG | With RAG | Status |
|-------|-------------|----------|--------|
| Canberra founding | 1908 (incorrect) | 1913 (correct) | ✅ Fixed |
| Retrieval latency | - | ~0.68s | ⚡ Fast |

## 🛠️ Advanced Usage

### Training from Scratch
```bash
# Full training (if you want to retrain)
python src\train_qlora.py

# Memory-optimized training
python src\train_qlora_min.py
```

### Custom Knowledge Base
```bash
# Add facts to RAG knowledge base
cd kb
python add_fact.py
```

### Evaluation & Analysis
```bash
# Generate evaluation summary
python src\summarize_eval.py

# View detailed results
type outputs\FINAL_REPORT.md
type outputs\EVAL_SUMMARY.json
```

## 📦 Release Package

Generate complete release bundle:
```bash
scripts\make_release.bat
# Creates: release_qwen25_qlora.zip
```

**Package Contents:**
- Trained adapter (`checkpoints/`)
- Portable inference tools (`portable_infer/`)
- Knowledge base (`kb/`)
- Evaluation results (`outputs/`)
- Documentation & scripts
- MIT License with third-party attributions (`LICENSE`)

## 🎬 Quick Demo

To demonstrate the QLoRA fine-tuning and RAG capabilities:
1. Launch the web UI: `cd portable_infer && python gradio_app.py`
2. Test without RAG: Ask "When was Canberra founded?" (expect: 1908)
3. Enable RAG toggle and ask again (expect: 1913 - correct answer)
4. Compare response quality and factual accuracy

## 🔧 Technical Details

### Memory Optimization
- **4-bit NF4 quantization** (BitsAndBytesConfig)
- **LoRA r=8** (attention layers only)
- **Gradient checkpointing** (trade compute for memory)
- **Sequence length 448** (vs 1024+ typical)
- **Gradient accumulation 32** (effective batch size)

### Hardware Requirements
- **Minimum**: 4GB VRAM (RTX 3050, RTX 4060)
- **Recommended**: 6GB+ VRAM for faster training
- **CPU**: Any modern CPU (RAG uses CPU embeddings)
- **RAM**: 8GB+ system RAM

### Dependencies
- PyTorch 2.1+ with CUDA support
- Transformers, PEFT, TRL, BitsAndBytesConfig
- FAISS (CPU), sentence-transformers
- Gradio for web interface

## 🚀 Production Deployment

### CLI Deployment
```bash
# Copy portable_infer/ to target machine
# Install: pip install -r portable_infer/requirements.txt
# Run: python portable_infer/run.py --adapter checkpoints/qwen25-3b-dolly-qlora-steps150
```

### Web UI Deployment
```bash
# Launch web interface
cd portable_infer
python gradio_app.py
# Access: http://localhost:7860
```

## 📈 Future Improvements

- **Extended Training**: +60 steps on higher VRAM hardware
- **Domain-Specific KB**: Expand knowledge base for specific use cases
- **Multi-GPU**: Scale training across multiple GPUs
- **Quantization**: Explore INT8/INT4 deployment optimizations

## 📄 License & Citation

This project builds upon:
- **Qwen2.5-3B-Instruct** (Alibaba Cloud)
- **Dolly-15k Dataset** (Databricks)
- **QLoRA Method** (Dettmers et al.)
- **PEFT Library** (Hugging Face)

---

**🎉 Ready for Production!** Complete QLoRA fine-tuning with RAG integration, optimized for consumer hardware.