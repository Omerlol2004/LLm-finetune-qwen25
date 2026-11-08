"""Centralized 4-bit QLoRA model loader, hardware utilities, and generation engine.

Consolidates duplicated ``bnb4()``, tokenizer setup, and generation boilerplate
previously scattered across 7 scripts, while adding:
- Root-anchored default paths (``PROJECT_ROOT``, ``DEFAULT_ADAPTER_DIR``, ``DEFAULT_KB_DIR``)
- Single-model evaluation support via ``disable_adapter=True`` (``with model.disable_adapter():``)
  to cut evaluation VRAM consumption in half on 6 GB GPUs.
- Distinct ``<|endoftext|>`` pad token vs ``<|im_end|>`` EOS token handling for training.
"""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_ADAPTER_DIR = Path(
    os.environ.get(
        "QWEN_ADAPTER_DIR",
        str(PROJECT_ROOT / "checkpoints" / "qwen25-3b-dolly-qlora-steps150"),
    )
)
DEFAULT_KB_DIR = Path(os.environ.get("QWEN_KB_DIR", str(PROJECT_ROOT / "kb")))
DEFAULT_SYSTEM_PROMPT = "You are a concise, friendly assistant. Use plain, factual language."


def bf16_supported() -> bool:
    """Return True if CUDA is available and supports native bfloat16."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        return False


def compute_dtype() -> Any:
    """Return optimal compute dtype (``torch.bfloat16`` or ``torch.float16``)."""
    import torch

    return torch.bfloat16 if bf16_supported() else torch.float16


def device_map_single() -> Dict[str, Union[int, str]]:
    """Pin all layers to CUDA device 0 when available to avoid split CPU/GPU placement."""
    try:
        import torch

        return {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    except ImportError:
        return {"": "cpu"}


def vram_status() -> str:
    """Return human-readable CUDA free/total VRAM status string."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return f"{free // (1024 ** 2)} / {total // (1024 ** 2)} MB free/total"
    except Exception:
        pass
    return "CUDA not available"


def build_bnb4_config(enable_cpu_offload: bool = False) -> Any:
    """Build 4-bit NormalFloat4 (NF4) double-quantization configuration."""
    from transformers import BitsAndBytesConfig

    kwargs: Dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": compute_dtype(),
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    if enable_cpu_offload:
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**kwargs)


def load_tokenizer(model_id: str = DEFAULT_BASE_MODEL, for_training: bool = False) -> Any:
    """Load Qwen2.5 tokenizer with safe pad/EOS token separation."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if for_training and "<|endoftext|>" in tok.get_vocab():
        # Preserve distinct pad token (<|endoftext|>) vs turn-end EOS (<|im_end|>)
        tok.pad_token = "<|endoftext|>"
        tok.pad_token_id = tok.convert_tokens_to_ids("<|endoftext|>")
        tok.padding_side = "right"
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_qwen_model(
    base_id: str = DEFAULT_BASE_MODEL,
    adapter_dir: Optional[Union[str, Path]] = DEFAULT_ADAPTER_DIR,
    load_adapter: bool = True,
) -> Tuple[Any, Any]:
    """Load 4-bit quantized base model and optionally attach QLoRA adapter weights."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = compute_dtype()
    dev_map = device_map_single()
    tok = load_tokenizer(base_id, for_training=False)

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map=dev_map,
        quantization_config=build_bnb4_config(),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model: Any = base
    if load_adapter and adapter_dir is not None:
        resolved_adapter = Path(adapter_dir)
        if not resolved_adapter.is_absolute():
            candidate = PROJECT_ROOT / resolved_adapter
            if candidate.exists():
                resolved_adapter = candidate
        if resolved_adapter.exists():
            model = PeftModel.from_pretrained(
                base,
                str(resolved_adapter),
                device_map=dev_map,
                torch_dtype=dtype,
            )

    model.config.use_cache = True
    model.eval()
    return tok, model


def generate_from_messages(
    tok: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    disable_adapter: bool = False,
) -> str:
    """Generate assistant response from a ChatML message list.

    When ``disable_adapter=True`` and ``model`` is a ``PeftModel``, temporarily
    disables LoRA adapters via ``with model.disable_adapter():`` so base-vs-adapter
    evaluations require only a single 4-bit base model in GPU memory.
    """
    import torch

    inputs = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    attn = torch.ones_like(inputs)
    inputs = inputs.to(model.device)
    attn = attn.to(model.device)

    do_sample = float(temperature) > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": tok.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    adapter_ctx = (
        model.disable_adapter()
        if (disable_adapter and hasattr(model, "disable_adapter"))
        else nullcontext()
    )

    with adapter_ctx, torch.inference_mode():
        out = model.generate(inputs, attention_mask=attn, **gen_kwargs)

    return tok.decode(out[0][inputs.shape[-1] :], skip_special_tokens=True).strip()


def generate_response(
    tok: Any,
    model: Any,
    system: str,
    user: str,
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    disable_adapter: bool = False,
) -> str:
    """Generate a response for a ``(system, user)`` prompt pair."""
    messages: List[Dict[str, str]] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return generate_from_messages(
        tok,
        model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        disable_adapter=disable_adapter,
    )
