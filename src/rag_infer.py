"""Standalone RAG inference demo with Confidence-Gated Retrieval.

Fixes over legacy implementation:
- Uses centralized ``src/engine.py`` and ``src/rag_utils.py`` instead of duplicating
  FAISS and ``BitsAndBytesConfig`` boilerplate.
- Bounds ``top_k <= index.ntotal`` and enforces ``--score_threshold`` (default ``0.30``)
  so irrelevant passages and ``-1`` FAISS sentinel indices are never injected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_KB_DIR,
    build_bnb4_config,
    generate_response,
    load_qwen_model,
)
from rag_utils import (  # noqa: E402
    build_rag_prompt,
    context_block,
    init_kb,
    retrieve as retrieve_kb,
)

BASE = DEFAULT_BASE_MODEL
ADAPTER_DIR = str(DEFAULT_ADAPTER_DIR)
KB_DIR = str(DEFAULT_KB_DIR)


def bnb4():
    """Return 4-bit NF4 quantization config."""
    return build_bnb4_config()


def load_llm(base_id: str = BASE, adapter_dir: str = ADAPTER_DIR):
    """Load 4-bit base model and QLoRA adapter."""
    return load_qwen_model(base_id=base_id, adapter_dir=adapter_dir, load_adapter=True)


def answer(tok, model, question: str, contexts, strict_kb: bool = False) -> str:
    """Generate a grounded answer from retrieved context passages."""
    if contexts:
        ctx_block = context_block(contexts)
        system = (
            "You answer using the CONTEXT below. "
            "If the answer is not in the context, say 'I don't know' briefly. "
            "Be factual and concise."
        )
        user = f"CONTEXT:\n{ctx_block}\n\nQUESTION: {question}"
    elif strict_kb:
        system = "Briefly state that the answer is not verified in the local knowledge base."
        user = question
    else:
        system = "You are a concise, factual assistant."
        user = question

    return generate_response(
        tok,
        model,
        system=system,
        user=user,
        max_new_tokens=160,
        temperature=0.2,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Confidence-Gated RAG inference")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    ap.add_argument("--kb_dir", default=KB_DIR)
    ap.add_argument(
        "--query",
        default="What is the capital of Australia and when was it founded?",
    )
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--score_threshold", type=float, default=0.30)
    ap.add_argument("--strict_kb", action="store_true")
    args = ap.parse_args(argv)

    tok, model = load_llm(args.base, args.adapter)
    init_kb(args.kb_dir)

    rag_plan = build_rag_prompt(
        system_prompt="You are a concise, factual assistant.",
        user_query=args.query,
        use_rag=True,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        strict_kb=args.strict_kb,
        kb_dir=args.kb_dir,
    )
    ctx = retrieve_kb(
        args.query,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        kb_dir=args.kb_dir,
    )

    print(f"\n--- RETRIEVED CONTEXT (mode={rag_plan['mode']}, tau={args.score_threshold:.2f}) ---")
    if ctx:
        for c in ctx:
            print(f"[{c['rank']}] {c['title']} (id={c.get('id', 'kb')}, score={c.get('score', 0.0):.4f})")
    else:
        print("(no passages exceeded score threshold)")

    t0 = time.time()
    ans = answer(tok, model, args.query, ctx, strict_kb=args.strict_kb)
    print("\n--- ANSWER ---\n", ans)
    print(f"\nLatency: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()