from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, time, os

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def vram():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return f"{free//(1024**2)} / {total//(1024**2)} MB free/total"
    return "CUDA not available"

def build_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def main():
    print("CUDA:", torch.cuda.is_available(), "| VRAM:", vram())
    print("Loading tokenizer/model:", MODEL_ID)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    bnb_cfg = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "Give me three bullet tips to stay focused while studying."},
    ]
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=160,    # if OOM, try 96 or 64
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
    )

    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(inputs, **gen_kwargs)
    dt = time.time() - t0

    gen_text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    print("\n--- MODEL OUTPUT ---\n", gen_text)
    print(f"\nLatency: {dt:.2f}s | Tokens generated: {out.shape[-1]-inputs.shape[-1]}")
    print("VRAM after generate:", vram())

if __name__ == "__main__":
    main()