"""Diagnostic benchmark for the 'Canberra 1913' factual hallucination failure mode.

Compares parametric-only QLoRA generation (which hallucinates 1901 Federation
or 1908 Yass-Canberra site selection) against Confidence-Gated RAG grounding
(which retrieves ``au_canberra_1913`` with cosine score > 0.30 and answers 1913).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
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
from rag_utils import build_rag_prompt  # noqa: E402

BASE = DEFAULT_BASE_MODEL
ADAPTER_DIR = str(DEFAULT_ADAPTER_DIR)


def bnb4():
    """Return 4-bit NF4 quantization config."""
    return build_bnb4_config()


def load_models(base_id: str = BASE, adapter_dir: str = ADAPTER_DIR):
    """Load 4-bit base model and QLoRA adapter."""
    return load_qwen_model(base_id=base_id, adapter_dir=adapter_dir, load_adapter=True)


def generate(tok, model, prompt: str, system: str = "") -> str:
    """Generate assistant response for ``prompt``."""
    return generate_response(
        tok,
        model,
        system=system,
        user=prompt,
        max_new_tokens=160,
        temperature=0.2,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Canberra 1913 parametric vs Confidence-Gated RAG test")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    ap.add_argument("--kb_dir", default=str(DEFAULT_KB_DIR))
    ap.add_argument("--score_threshold", type=float, default=0.30)
    args = ap.parse_args(argv)

    tok, model = load_models(args.base, args.adapter)
    question = "What is the capital of Australia and when was it founded?"

    print(f"\n--- QUESTION (NO RAG) ---\n{question}")
    t0 = time.time()
    answer_no_rag = generate(tok, model, question)
    print(f"\n--- ANSWER (NO RAG) ---\n{answer_no_rag}")
    print(f"Latency: {time.time() - t0:.2f}s")

    rag_plan = build_rag_prompt(
        system_prompt="You are a concise, factual assistant.",
        user_query=question,
        use_rag=True,
        top_k=3,
        score_threshold=args.score_threshold,
        strict_kb=False,
        kb_dir=args.kb_dir,
    )
    print(f"\n--- QUESTION (CONFIDENCE-GATED RAG, tau={args.score_threshold:.2f}) ---")
    print("Retrieved:", rag_plan["retrieved_summary"])
    t1 = time.time()
    answer_rag = generate(tok, model, rag_plan["user"], system=rag_plan["system"])
    print(f"\n--- ANSWER (WITH RAG) ---\n{answer_rag}")
    print(f"Latency: {time.time() - t1:.2f}s")


if __name__ == "__main__":
    main()