"""Interactive CLI runner for Qwen2.5-3B + QLoRA + Confidence-Gated RAG.

Upgrades over legacy implementation:
- Root-anchored default ``--adapter`` and ``--kb_dir`` paths so running from
  either repository root or ``portable_infer/`` works out-of-the-box.
- Lazy KB initialization only when ``--rag`` is enabled.
- ``--score_threshold`` (default ``0.30``) and ``--strict_kb`` flags for
  confidence-gated hybrid or strict retrieval.
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
    DEFAULT_SYSTEM_PROMPT,
    build_bnb4_config,
    generate_response,
    load_qwen_model,
)
from rag_utils import build_rag_prompt, init_kb  # noqa: E402


def bnb4():
    """Backward-compatible helper returning the 4-bit NF4 configuration."""
    return build_bnb4_config()


def load_models(base_id: str, adapter_dir: str):
    """Load 4-bit Qwen2.5 base model and QLoRA adapter."""
    return load_qwen_model(base_id=base_id, adapter_dir=adapter_dir, load_adapter=True)


def generate(tok, model, system: str, user: str, max_new_tokens: int = 160, temperature: float = 0.3) -> str:
    """Generate assistant response for a system and user prompt."""
    return generate_response(
        tok,
        model,
        system=system,
        user=user,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Interactive Qwen2.5-3B QLoRA + Confidence-Gated RAG CLI")
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER_DIR))
    ap.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--rag", action="store_true", help="Enable confidence-gated retrieval from local KB")
    ap.add_argument("--kb_dir", default=str(DEFAULT_KB_DIR))
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument(
        "--score_threshold",
        type=float,
        default=0.30,
        help="Minimum cosine similarity tau required to inject retrieved context (default: 0.30)",
    )
    ap.add_argument(
        "--strict_kb",
        action="store_true",
        help="Abstain when no KB passage meets --score_threshold instead of falling back to general knowledge",
    )
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.3)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tok, model = load_models(args.base, args.adapter)
    if args.rag:
        info = init_kb(args.kb_dir)
        print(
            f"RAG enabled (kb={info['kb_dir']}, backend={info['backend']}, "
            f"top_k={args.top_k}, tau={args.score_threshold:.2f}, strict={args.strict_kb})"
        )

    print("Loaded. Type your message (or 'exit').\n")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if not user or user.lower() in {"exit", "quit"}:
            break

        rag_plan = build_rag_prompt(
            system_prompt=args.system,
            user_query=user,
            use_rag=args.rag,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            strict_kb=args.strict_kb,
            kb_dir=args.kb_dir,
        )
        if args.rag:
            print(f"Retrieved [{rag_plan['mode']}]: {rag_plan['retrieved_summary']}")

        t0 = time.time()
        ans = generate(
            tok,
            model,
            rag_plan["system"],
            rag_plan["user"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nAssistant: {ans}\n(Gen time: {time.time() - t0:.2f}s)\n")


if __name__ == "__main__":
    main()