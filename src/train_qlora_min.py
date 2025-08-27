import os, json, math, time, argparse, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training

def bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def bnb4():
    return BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=(torch.bfloat16 if bf16() else torch.float16),
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True)

def render_chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def tokenize_batch(tok, texts, max_len):
    toks = tok(texts, max_length=max_len, truncation=True, padding=False)
    # labels = input_ids (causal LM); will be prepared by collator
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_jsonl", default="data/processed/train_patch.jsonl")
    ap.add_argument("--val_jsonl", default="data/processed/val.jsonl")
    ap.add_argument("--output_dir", default="checkpoints/qwen25-3b-dolly-qlora-steps210_patch")
    ap.add_argument("--init_from_adapter", default="checkpoints/qwen25-3b-dolly-qlora-steps150")
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=30)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if bf16() else torch.float16

    print(">>> Loading tokenizer:", args.model_id)
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    print(">>> Loading base model in 4-bit …")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", quantization_config=bnb4(),
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(base)

    # Warm-start from existing adapter (no Trainer resume)
    if args.init_from_adapter and os.path.isdir(args.init_from_adapter):
        print(">>> Loading existing LoRA adapters:", args.init_from_adapter)
        model = PeftModel.from_pretrained(base, args.init_from_adapter, is_trainable=True)
    else:
        print(">>> Creating fresh LoRA adapters")
        lora = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)

    print(">>> Loading datasets …")
    ds = load_dataset("json", data_files={"train": args.train_jsonl, "val": args.val_jsonl})

    def to_text(batch):
        texts = []
        for msgs in batch["messages"]:
            texts.append(render_chat(tok, msgs))
        toks = tokenize_batch(tok, texts, args.max_seq_len)
        return toks

    cols_to_remove = [c for c in ds["train"].column_names if c != "messages"]
    train_ds = ds["train"].map(to_text, batched=True, remove_columns=cols_to_remove)
    val_ds   = ds["val"].map(to_text, batched=True, remove_columns=cols_to_remove)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    steps_per_epoch = math.ceil(len(train_ds) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    print(f">>> Train samples: {len(train_ds)} | steps/epoch≈{steps_per_epoch}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=10**9,              # avoid mid-run saves
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
        tokenizer=tok,
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