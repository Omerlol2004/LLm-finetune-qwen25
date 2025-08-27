import os, json, time, torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "checkpoints/qwen25-3b-dolly-qlora-steps150"
KB_DIR = "kb"
INDEX_PATH = os.path.join(KB_DIR, "index.faiss")
META_PATH  = os.path.join(KB_DIR, "meta.json")

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_llm():
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

def load_kb():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH,"r",encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return index, meta, embedder

def retrieve(embedder, index, meta, query, k=3):
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(q.astype("float32"), k)
    hits = []
    for rank, idx in enumerate(I[0]):
        title = meta["titles"][idx]; text = meta["texts"][idx]
        hits.append({"rank":rank+1,"title":title,"text":text})
    return hits

def answer(tok, model, question, contexts):
    ctx_block = "\n\n".join([f"[{c['rank']}] {c['title']}: {c['text']}" for c in contexts])
    system = (
        "You answer using the CONTEXT below. "
        "If the answer is not in the context, say 'I don't know' briefly. "
        "Be factual and concise."
    )
    user = f"CONTEXT:\n{ctx_block}\n\nQUESTION: {question}"
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
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
    tok, model = load_llm()
    index, meta, embedder = load_kb()

    q = "What is the capital of Australia and when was it founded?"
    ctx = retrieve(embedder, index, meta, q, k=3)
    print("\n--- RETRIEVED CONTEXT ---")
    for c in ctx: print(f"[{c['rank']}] {c['title']}")
    t0=time.time()
    ans = answer(tok, model, q, ctx)
    print("\n--- ANSWER ---\n", ans)
    print(f"\nLatency: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()