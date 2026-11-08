"""Primary QLoRA Supervised Fine-Tuning (SFT) trainer for Qwen2.5-3B-Instruct.

Fixes over legacy implementation:
- Properly wires ``--max_seq_len`` and ``--packing`` (with ``BooleanOptionalAction``
  so ``--no-packing`` works) into ``SFTConfig`` / ``SFTTrainer``.
- Preserves distinct ``<|endoftext|>`` pad token vs ``<|im_end|>`` turn-end EOS token.
- Adds ``--target_all_linear`` flag to optionally include MLP projections
  (``gate_proj``, ``up_proj``, ``down_proj``) alongside attention projections.
"""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
import sys
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    bf16_supported,
    build_bnb4_config,
    compute_dtype,
    device_map_single,
    load_tokenizer,
)

ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_target_modules(target_all_linear: bool = False) -> List[str]:
    """Return LoRA target module list for attention-only or all-linear adapters."""
    return list(ALL_LINEAR_MODULES if target_all_linear else ATTENTION_MODULES)


def load_jsonl(train_path: str, val_path: str):
    """Load train and validation JSONL splits via Hugging Face ``datasets``."""
    from datasets import load_dataset

    data_files = {"train": str(train_path), "validation": str(val_path)}
    return load_dataset("json", data_files=data_files)


def format_messages(tokenizer: Any, example: dict) -> str:
    """Convert ``{"messages": [...]}`` into ChatML text."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT trainer for Qwen2.5-3B-Instruct")
    parser.add_argument("--model_id", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "val.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "qwen25-3b-dolly-qlora"),
    )
    parser.add_argument("--max_seq_len", type=int, default=448)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack multiple short examples into a single sequence (--packing / --no-packing)",
    )
    parser.add_argument(
        "--target_all_linear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach LoRA to both attention and SwiGLU MLP projections (q/k/v/o/gate/up/down)",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--init_from_adapter", type=str, default=None)
    return parser


def main(argv=None):
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer

    args = build_parser().parse_args(argv)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = compute_dtype()

    print(">>> Loading tokenizer:", args.model_id)
    tok = load_tokenizer(args.model_id, for_training=True)

    print(">>> Loading base model in 4-bit …")
    bnb_cfg = build_bnb4_config(enable_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map_single(),
        quantization_config=bnb_cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_cfg = None
    if args.init_from_adapter:
        model = PeftModel.from_pretrained(
            model,
            args.init_from_adapter,
            is_trainable=True,
        )
    else:
        target_modules = resolve_target_modules(args.target_all_linear)
        print(f">>> Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha}, modules={target_modules})")
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    print(">>> Loading dataset …")
    ds = load_jsonl(args.train_jsonl, args.val_jsonl)

    def formatting_func(examples):
        texts = [format_messages(tok, {"messages": msgs}) for msgs in examples["messages"]]
        return {"text": texts}

    ds = ds.map(formatting_func, batched=True, remove_columns=ds["train"].column_names)

    total_train_examples = len(ds["train"])
    steps_per_epoch = math.ceil(
        total_train_examples / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    print(
        f">>> Train examples: {total_train_examples} | steps/epoch≈{steps_per_epoch} "
        f"| max_seq_len={args.max_seq_len} | packing={args.packing}"
    )

    common_targs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16_supported(),
        fp16=not bf16_supported(),
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to=["tensorboard"],
        save_safetensors=True,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
    }
    if peft_cfg is not None:
        trainer_kwargs["peft_config"] = peft_cfg

    # Support both modern TRL (SFTConfig) and older TRL (SFTTrainer kwargs)
    try:
        from trl import SFTConfig

        sft_sig = inspect.signature(SFTConfig.__init__).parameters
        sft_extra: dict[str, Any] = {"packing": bool(args.packing)}
        if "max_seq_length" in sft_sig:
            sft_extra["max_seq_length"] = int(args.max_seq_len)
        elif "max_length" in sft_sig:
            sft_extra["max_length"] = int(args.max_seq_len)
        if "dataset_text_field" in sft_sig:
            sft_extra["dataset_text_field"] = "text"
        targs = SFTConfig(**common_targs, **sft_extra)
    except ImportError:
        targs = TrainingArguments(**common_targs)
        trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
        if "max_seq_length" in trainer_sig:
            trainer_kwargs["max_seq_length"] = int(args.max_seq_len)
        if "packing" in trainer_sig:
            trainer_kwargs["packing"] = bool(args.packing)
        if "dataset_text_field" in trainer_sig:
            trainer_kwargs["dataset_text_field"] = "text"

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tok

    trainer = SFTTrainer(args=targs, **trainer_kwargs)

    print(">>> Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(">>> Training complete. Saving adapter weights …")
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(">>> Done. Adapters saved to:", args.output_dir)


if __name__ == "__main__":
    main()