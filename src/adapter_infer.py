from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, time

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "checkpoints/qwen25-3b-dolly-qlora-steps150"  # change if needed

def dev():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_adapter():
    d = dev()
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Force everything onto a single CUDA device to avoid split (CPU) placement
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map={"":0} if d.startswith("cuda") else {"":d},
        quantization_config=bnb4(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        device_map={"":0} if d.startswith("cuda") else {"":d},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model.config.use_cache = True
    model.eval()
    return tok, model

def chat(prompt):
    tok, model = load_adapter()

    messages = [
        {"role":"system","content":"You are a concise, helpful assistant."},
        {"role":"user","content":prompt},
    ]
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # Explicit attention mask (no padding in single prompt → all ones)
    attn = torch.ones_like(inputs)

    inputs = inputs.to(model.device)
    attn = attn.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=160,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
    )

    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(inputs, attention_mask=attn, **gen_kwargs)
    dt = time.time() - t0

    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    print("\n--- ANSWER ---\n", text)
    print(f"\nLatency: {dt:.2f}s")

if __name__ == "__main__":
    tests = [
        "What is the capital of Australia and when was it founded?",
        "Explain the difference between precision and recall with a short example.",
        "Rewrite this in friendlier tone: 'Your request was denied due to policy violations.'",
        "Give three study techniques with one pro and one con each.",
    ]
    for p in tests:
        print("\n======================\nPROMPT:", p)
        chat(p)