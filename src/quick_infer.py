"""Hardware & 4-bit base model smoke test for 6 GB VRAM GPUs.

Fixes over legacy implementation:
- Uses dynamic ``bf16_supported()`` instead of hardcoded ``torch.bfloat16`` so
  Turing/Volta GPUs (T4, RTX 20-series) fall back cleanly to ``torch.float16``.
- Explicitly constructs and passes ``attention_mask`` and ``pad_token_id`` to
  ``model.generate()``.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    bf16_supported,
    build_bnb4_config,
    generate_response,
    load_qwen_model,
    vram_status,
)

MODEL_ID = DEFAULT_BASE_MODEL


def vram() -> str:
    """Return free/total CUDA VRAM summary string."""
    return vram_status()


def build_bnb_config():
    """Build 4-bit NF4 quantization config with dynamic bf16/fp16 selection."""
    return build_bnb4_config()


def main():
    import torch

    print(
        "CUDA:",
        torch.cuda.is_available(),
        "| bf16:",
        bf16_supported(),
        "| VRAM:",
        vram(),
    )
    print("Loading tokenizer/model:", MODEL_ID)

    tok, model = load_qwen_model(MODEL_ID, adapter_dir=None, load_adapter=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    gen_text = generate_response(
        tok,
        model,
        system="You are a concise, helpful assistant.",
        user="Give me three bullet tips to stay focused while studying.",
        max_new_tokens=160,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    dt = time.time() - t0

    print("\n--- MODEL OUTPUT ---\n", gen_text)
    print(f"\nLatency: {dt:.2f}s")
    print("VRAM after generate:", vram())


if __name__ == "__main__":
    main()