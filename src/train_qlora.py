import os, argparse, math, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

def bf16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def load_jsonl(train_path, val_path):
    data_files = {"train": train_path, "validation": val_path}
    ds = load_dataset("json", data_files=data_files)
    return ds

def format_messages(tokenizer, example):
    # Convert {"messages": [...]} → chat template text
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_jsonl", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen25-3b-dolly-qlora")
    parser.add_argument("--max_seq_len", type=int, default=448)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--packing", action="store_true", default=True)  # pack multiple samples per sequence
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--init_from_adapter", type=str, default=None)
    args = parser.parse_args()

    # Speed/precision knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if bf16_supported() else torch.float16

    print(">>> Loading tokenizer:", args.model_id)
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    print(">>> Loading base model in 4-bit …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"":0},
        quantization_config=bnb_cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # crucial for memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # Prepare model for k-bit training + attach LoRA
    model = prepare_model_for_kbit_training(model)
    
    peft_cfg = None
    if args.init_from_adapter:
        # Load existing LoRA adapters and keep them trainable
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.init_from_adapter,
            is_trainable=True,
        )
    else:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],  # attention only
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )

    print(">>> Loading dataset …")
    ds = load_jsonl(args.train_jsonl, args.val_jsonl)

    # map → text (leave tokenization to Trainer)
    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            texts.append(format_messages(tok, {"messages": msgs}))
        return {"text": texts}

    ds = ds.map(formatting_func, batched=True, remove_columns=ds["train"].column_names)

    # TrainingArguments
    total_train_examples = len(ds["train"])
    steps_per_epoch = math.ceil(total_train_examples / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    print(f">>> Train examples: {total_train_examples} | steps/epoch≈{steps_per_epoch}")

    targs = TrainingArguments(
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=targs,
        **({"peft_config": peft_cfg} if peft_cfg is not None else {})
    )

    print(">>> Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(">>> Training complete. Saving adapter weights …")
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(">>> Done. Adapters saved to:", args.output_dir)

if __name__ == "__main__":
    main()