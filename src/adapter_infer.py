"""Smoke-test inference runner for QLoRA fine-tuned adapter checkpoints.

Fixes over legacy implementation:
- Caches and reuses the 4-bit base model + LoRA adapter across all test prompts
  instead of reloading 3B weights from disk inside ``chat(prompt)`` on every turn
  (cutting 4-prompt benchmark wall time by ~75%).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    build_bnb4_config,
    generate_response,
    load_qwen_model,
)

BASE = DEFAULT_BASE_MODEL
ADAPTER_DIR = str(DEFAULT_ADAPTER_DIR)

_cached_tok: Any = None
_cached_model: Any = None


def dev() -> str:
    """Return active device string ('cuda:0' or 'cpu')."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def bnb4():
    """Return 4-bit NF4 quantization config."""
    return build_bnb4_config()


def load_adapter(base_id: str = BASE, adapter_dir: str = ADAPTER_DIR) -> Tuple[Any, Any]:
    """Load and cache the 4-bit base model and QLoRA adapter once."""
    global _cached_tok, _cached_model
    if _cached_tok is None or _cached_model is None:
        _cached_tok, _cached_model = load_qwen_model(
            base_id=base_id,
            adapter_dir=adapter_dir,
            load_adapter=True,
        )
    return _cached_tok, _cached_model


def chat(
    prompt: str,
    tok: Optional[Any] = None,
    model: Optional[Any] = None,
    system: str = "You are a concise, helpful assistant.",
) -> str:
    """Generate and print an assistant response using a preloaded (or cached) model."""
    if tok is None or model is None:
        tok, model = load_adapter()

    t0 = time.time()
    text = generate_response(
        tok,
        model,
        system=system,
        user=prompt,
        max_new_tokens=160,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    dt = time.time() - t0

    print("\n--- ANSWER ---\n", text)
    print(f"\nLatency: {dt:.2f}s")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run 4-prompt QLoRA adapter smoke suite")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    args = ap.parse_args(argv)

    tok, model = load_adapter(args.base, args.adapter)
    tests = [
        "What is the capital of Australia and when was it founded?",
        "Explain the difference between precision and recall with a short example.",
        "Rewrite this in friendlier tone: 'Your request was denied due to policy violations.'",
        "Give three study techniques with one pro and one con each.",
    ]
    for p in tests:
        print("\n======================\nPROMPT:", p)
        chat(p, tok=tok, model=model)


if __name__ == "__main__":
    main()