import argparse, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rag_utils import init_kb, retrieve, context_block

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_models(base_id, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        quantization_config=bnb4(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base, adapter_dir,
        device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = True
    model.eval()
    return tok, model

def generate(tok, model, system, user, max_new_tokens=160, temperature=0.5):
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    attn = torch.ones_like(inputs)
    inputs, attn = inputs.to(model.device), attn.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs, attention_mask=attn,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.9,
            repetition_penalty=1.05, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../checkpoints/qwen25-3b-dolly-qlora-steps150")
    ap.add_argument("--system", default="You are a concise, friendly assistant. Use plain, factual language.")
    ap.add_argument("--rag", action="store_true", help="Enable retrieval from local KB")
    ap.add_argument("--kb_dir", default="../kb")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    tok, model = load_models(args.base, args.adapter)
    if args.rag:
        init_kb(args.kb_dir)
        print(f"RAG enabled (kb={args.kb_dir}, top_k={args.top_k})")

    print("Loaded. Type your message (or 'exit').\n")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if not user or user.lower() in {"exit","quit"}:
            break

        user_text = user
        if args.rag:
            hits = retrieve(user_text, top_k=args.top_k)
            print("Retrieved:", ", ".join([h["title"] for h in hits]))
            ctx = context_block(hits)
            system = "Answer using the CONTEXT. If not in context, say 'I don't know'. Be concise."
            user_text = f"CONTEXT:\n{ctx}\n\nQUESTION: {user}"

        t0=time.time()
        ans = generate(tok, model, args.system if not args.rag else system, user_text,
                       max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print(f"\nAssistant: {ans}\n(Gen time: {time.time()-t0:.2f}s)\n")

if __name__ == "__main__":
    main()