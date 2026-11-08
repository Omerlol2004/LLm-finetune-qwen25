"""Gradio Web UI for Qwen2.5-3B + QLoRA + Confidence-Gated RAG.

Fixes over legacy implementation:
- Root-anchored ``ADAPTER`` and ``KB_DIR`` paths so launching from repo root or
  ``portable_infer/`` works identically.
- Word-boundary greeting check via ``build_rag_prompt`` BEFORE FAISS retrieval,
  fixing the critical bug where ``"hi"`` matched ``"which"``, ``"this"``, and ``"history"``.
- Interactive ``Score Threshold (tau)`` slider (default ``0.30``) and ``Strict KB`` toggle
  displaying retrieved passage IDs and cosine similarity scores.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_KB_DIR,
    DEFAULT_SYSTEM_PROMPT,
    generate_response,
    load_qwen_model,
)
from rag_utils import build_rag_prompt, init_kb  # noqa: E402

BASE = DEFAULT_BASE_MODEL
ADAPTER = str(DEFAULT_ADAPTER_DIR)
KB_DIR = str(DEFAULT_KB_DIR)

_tok: Any = None
_model: Any = None


def load() -> str:
    """Lazy-load 4-bit base model and QLoRA adapter on startup or first request."""
    global _tok, _model
    if _tok is not None and _model is not None:
        return "Model already loaded."
    _tok, _model = load_qwen_model(BASE, ADAPTER, load_adapter=True)
    return f"Model loaded: {BASE} + {Path(ADAPTER).name}"


def chat(
    system: str,
    user: str,
    use_rag: bool,
    top_k: int,
    score_threshold: float = 0.30,
    strict_kb: bool = False,
) -> Tuple[str, str, str]:
    """Run confidence-gated RAG or direct QLoRA chat inference."""
    if not user or not user.strip():
        return "Please enter a question.", "(empty query)", "0.00s"

    if _tok is None or _model is None:
        load()

    if use_rag:
        init_kb(KB_DIR)

    rag_plan = build_rag_prompt(
        system_prompt=system or DEFAULT_SYSTEM_PROMPT,
        user_query=user,
        use_rag=bool(use_rag),
        top_k=int(top_k),
        score_threshold=float(score_threshold),
        strict_kb=bool(strict_kb),
        kb_dir=KB_DIR,
    )

    t0 = time.time()
    text = generate_response(
        _tok,
        _model,
        system=rag_plan["system"],
        user=rag_plan["user"],
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    dt = time.time() - t0
    return text, rag_plan["retrieved_summary"], f"{dt:.2f}s"


def build_demo():
    """Construct the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(title="Qwen2.5-3B + QLoRA + Confidence-Gated RAG") as demo:
        gr.Markdown("# Qwen2.5-3B (QLoRA) — Confidence-Gated Hybrid RAG")
        status = gr.Markdown("Model loads automatically on startup or first query.")

        with gr.Row():
            system = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="System Prompt",
            )
            use_rag = gr.Checkbox(value=True, label="Use RAG (Local FAISS KB)")
            strict_kb = gr.Checkbox(value=False, label="Strict KB Only (Abstain if < τ)")

        with gr.Row():
            top_k = gr.Slider(1, 5, step=1, value=3, label="Top-K Passages")
            score_threshold = gr.Slider(
                0.0,
                0.8,
                step=0.05,
                value=0.30,
                label="Score Threshold (τ)",
            )

        user = gr.Textbox(lines=4, label="User Question")
        btn = gr.Button("Ask")
        out = gr.Textbox(lines=10, label="Assistant Response")
        with gr.Row():
            retrieved = gr.Textbox(label="Retrieved Passages (ID & Cosine Score)")
            latency = gr.Textbox(label="Generation Latency")

        btn.click(
            chat,
            inputs=[system, user, use_rag, top_k, score_threshold, strict_kb],
            outputs=[out, retrieved, latency],
        )
        demo.load(load, inputs=None, outputs=status)
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)