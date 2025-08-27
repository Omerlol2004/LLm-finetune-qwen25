# Portable Inference
- `pip install -r requirements.txt`
- CLI: `python run.py --adapter ../checkpoints/qwen25-3b-dolly-qlora-steps150`
- UI: `python gradio_app.py` → open http://localhost:7860
- Needs ~3GB free VRAM (4-bit). Base model downloads automatically.