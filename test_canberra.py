import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "checkpoints/qwen25-3b-dolly-qlora-steps150"

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_models():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        BASE, device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        quantization_config=bnb4(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base, ADAPTER_DIR,
        device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = True
    model.eval()
    return tok, model

def generate(tok, model, prompt):
    messages = [{"role":"user","content":prompt}]
    inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    attn = torch.ones_like(inputs)
    inputs, attn = inputs.to(model.device), attn.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs, attention_mask=attn, max_new_tokens=160, temperature=0.2, top_p=0.9,
            repetition_penalty=1.05, pad_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return text

def main():
    tok, model = load_models()
    question = "What is the capital of Australia and when was it founded?"
    print(f"\n--- QUESTION (NO RAG) ---\n{question}")
    t0 = time.time()
    answer = generate(tok, model, question)
    print(f"\n--- ANSWER (NO RAG) ---\n{answer}")
    print(f"\nLatency: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()