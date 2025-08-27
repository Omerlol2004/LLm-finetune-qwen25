@echo off
cd portable_infer
python run.py --adapter ..\checkpoints\qwen25-3b-dolly-qlora-steps150 --base Qwen/Qwen2.5-3B-Instruct --rag --kb_dir ..\kb --top_k 3