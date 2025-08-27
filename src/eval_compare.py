import os, json, random, time, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate, torch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qwen25-3b-dolly-qlora-steps150")
    ap.add_argument("--val_path", default="data/processed/val.jsonl")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    return ap.parse_args()

def device_map_single():
    return {"":0} if torch.cuda.is_available() else {"":"cpu"}

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_base(base_id):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map=device_map_single(), quantization_config=bnb4(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    model.eval()
    return tok, model

def load_adapter(base_id, adapter_dir):
    tok, base = load_base(base_id)
    model = PeftModel.from_pretrained(
        base, adapter_dir, device_map=device_map_single(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = True
    model.eval()
    return tok, model

def render(tok, messages, add_gen=False):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_gen, return_tensors="pt")

def gen_text(tok, model, messages, max_new_tokens):
    inputs = render(tok, messages, add_gen=True)
    attn = torch.ones_like(inputs)
    inputs = inputs.to(model.device); attn = attn.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs, attention_mask=attn,
            max_new_tokens=max_new_tokens, temperature=0.2, top_p=0.9,
            repetition_penalty=1.05, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

def main():
    args = parse_args()
    ds = load_dataset("json", data_files={"val": args.val_path})["val"]
    idxs = list(range(len(ds))); random.seed(42); random.shuffle(idxs); idxs = idxs[:args.num_samples]

    tok_b, base = load_base(args.base)
    tok_a, adap = load_adapter(args.base, args.adapter)
    rouge = evaluate.load("rouge")

    rows = []
    t0=time.time()
    for i,k in enumerate(idxs, 1):
        msgs = ds[k]["messages"]
        gold = next((m["content"] for m in msgs if m["role"]=="assistant"), "")
        user_msgs = [m for m in msgs if m["role"]!="assistant"]
        pred_base = gen_text(tok_b, base, user_msgs, args.max_new_tokens)
        pred_adap = gen_text(tok_a, adap, user_msgs, args.max_new_tokens)
        rows.append({"idx":k, "user":user_msgs[-1]["content"], "gold":gold, "base":pred_base, "adapter":pred_adap})
        if i%10==0: print(f"Processed {i}/{len(idxs)}")

    base_scores = rouge.compute(predictions=[r["base"] for r in rows], references=[r["gold"] for r in rows])
    adap_scores = rouge.compute(predictions=[r["adapter"] for r in rows], references=[r["gold"] for r in rows])

    dt=time.time()-t0
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_compare.jsonl","w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("\n=== ROUGE-L F1 ===")
    print("Base   :", round(base_scores['rougeLsum'],4))
    print("Adapter:", round(adap_scores['rougeLsum'],4))
    print(f"Wall time: {dt/60:.1f} min for {len(rows)} samples")
    print("Saved details to outputs/eval_compare.jsonl")

if __name__ == "__main__":
    main()