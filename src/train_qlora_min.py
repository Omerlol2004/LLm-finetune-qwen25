"""Minimal Trainer-based QLoRA continuation script for targeted factual/style patches.

Fixes over legacy implementation:
- Assistant completion-only label masking (``tokenize_with_assistant_mask``): sets
  system and user prompt token positions to ``-100`` so gradient updates focus
  100% on the assistant completion rather than memorizing system/user boilerplate.
- Pins model to ``device_map={"": 0}`` when CUDA is available (instead of ``"auto"``)
  to prevent ``accelerate`` from offloading 4-bit layers to CPU RAM under ``paged_adamw_8bit``.
- Restores ``max_grad_norm=0.3`` gradient clipping for stable continuation fine-tuning.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    bf16_supported,
    build_bnb4_config,
    compute_dtype,
    device_map_single,
    load_tokenizer,
)


def bf16() -> bool:
    """Return True if CUDA bfloat16 is supported."""
    return bf16_supported()


def bnb4():
    """Return 4-bit NF4 quantization configuration."""
    return build_bnb4_config()


def render_chat(tok: Any, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render ChatML message list into formatted text."""
    return tok.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def tokenize_with_assistant_mask(
    tok: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    max_len: int,
) -> Dict[str, List[List[int]]]:
    """Tokenize conversations and mask non-assistant prefix tokens with ``-100``.

    For each conversation ``[system, user, ..., assistant]``, tokens belonging to
    the prompt prefix ``messages[:-1]`` (with ``add_generation_prompt=True``) are
    assigned label ``-100`` so cross-entropy loss is computed exclusively on the
    assistant's target response.
    """
    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for msgs in messages_batch:
        full_text = render_chat(tok, msgs, add_generation_prompt=False)
        full_enc = tok(
            full_text,
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        ids = list(full_enc["input_ids"])
        attn = list(full_enc["attention_mask"])

        if len(msgs) > 1 and msgs[-1].get("role") == "assistant":
            prompt_text = render_chat(tok, msgs[:-1], add_generation_prompt=True)
            prompt_ids = tok(
                prompt_text,
                max_length=max_len,
                truncation=True,
                padding=False,
            )["input_ids"]
            prompt_len = min(len(prompt_ids), len(ids))
        else:
            prompt_len = 0

        labels = [-100] * prompt_len + ids[prompt_len:]
        input_ids_list.append(ids)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Completion-masked QLoRA micro-patch trainer")
    ap.add_argument("--model_id", default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--train_jsonl",
        default=str(PROJECT_ROOT / "data" / "processed" / "train_patch.jsonl"),
    )
    ap.add_argument(
        "--val_jsonl",
        default=str(PROJECT_ROOT / "data" / "processed" / "val.jsonl"),
    )
    ap.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "qwen25-3b-dolly-qlora-steps210_patch"),
    )
    ap.add_argument("--init_from_adapter", default=str(DEFAULT_ADAPTER_DIR))
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_grad_norm", type=float, default=0.3)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=30)
    return ap


def main(argv=None):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    args = build_parser().parse_args(argv)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = compute_dtype()

    print(">>> Loading tokenizer:", args.model_id)
    tok = load_tokenizer(args.model_id, for_training=True)

    print(">>> Loading base model in 4-bit …")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map_single(),
        quantization_config=bnb4(),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base)

    if args.init_from_adapter and os.path.isdir(args.init_from_adapter):
        print(">>> Loading existing LoRA adapters:", args.init_from_adapter)
        model = PeftModel.from_pretrained(base, args.init_from_adapter, is_trainable=True)
    else:
        print(">>> Creating fresh LoRA adapters")
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)

    print(">>> Loading datasets …")
    ds = load_dataset("json", data_files={"train": args.train_jsonl, "val": args.val_jsonl})

    def to_masked_tokens(batch):
        return tokenize_with_assistant_mask(tok, batch["messages"], args.max_seq_len)

    cols_to_remove = list(ds["train"].column_names)
    train_ds = ds["train"].map(to_masked_tokens, batched=True, remove_columns=cols_to_remove)
    val_ds = ds["val"].map(to_masked_tokens, batched=True, remove_columns=cols_to_remove)

    collator = DataCollatorForSeq2Seq(
        tok,
        padding=True,
        label_pad_token_id=-100,
    )

    steps_per_epoch = math.ceil(
        len(train_ds) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    print(f">>> Train samples: {len(train_ds)} | steps/epoch≈{steps_per_epoch}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=10 ** 9,
        save_total_limit=1,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16(),
        fp16=not bf16(),
        optim="paged_adamw_8bit",
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds.select(range(min(300, len(val_ds)))),
        data_collator=collator,
    )

    print(">>> Starting training …")
    trainer.train()
    print(">>> Training complete. Saving adapter …")
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(">>> Done:", args.output_dir)


if __name__ == "__main__":
    main()