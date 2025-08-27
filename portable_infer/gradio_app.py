import gradio as gr, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rag_utils import init_kb, retrieve, context_block

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "../checkpoints/qwen25-3b-dolly-qlora-steps150"
KB_DIR = "../kb"

def bnb4():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

_tok, _model = None, None
def load():
    global _tok, _model
    _tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
        _tok.pad_token_id = _tok.eos_token_id
    base_m = AutoModelForCausalLM.from_pretrained(
        BASE, device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        quantization_config=bnb4(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
    )
    _model = PeftModel.from_pretrained(
        base_m, ADAPTER,
        device_map={"":0} if torch.cuda.is_available() else {"":"cpu"},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    _model.config.use_cache = True
    _model.eval()
    return "Model loaded."

def chat(system, user, use_rag, top_k):
    if use_rag:
        init_kb(KB_DIR)
        hits = retrieve(user, top_k=top_k)
        ctx = context_block(hits)
        
        # Check if query is conversational/greeting (no factual retrieval needed)
        conversational_queries = ["hi", "hello", "hey", "how are you", "thanks", "thank you", "bye", "goodbye"]
        is_conversational = any(greeting in user.lower() for greeting in conversational_queries)
        
        if is_conversational:
            # For greetings, use normal conversation mode
            retrieved = "(conversational - no retrieval)"
        elif ctx.strip():
            # For queries with relevant context, enhance with RAG
            system = "You are a helpful assistant. Use the provided CONTEXT to enhance your answer when relevant, but you can also use your general knowledge. Be informative and concise."
            user = f"CONTEXT:\n{ctx}\n\nQUESTION: {user}"
            retrieved = ", ".join([h["title"] for h in hits])
        else:
            # For queries with no relevant context, answer normally
            retrieved = "(no relevant context - general knowledge)"
    else:
        retrieved = "(none)"

    tok_in = _tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    attn = torch.ones_like(tok_in)
    tok_in, attn = tok_in.to(_model.device), attn.to(_model.device)
    t0=time.time()
    with torch.inference_mode():
        out = _model.generate(tok_in, attention_mask=attn, max_new_tokens=256, temperature=0.6, top_p=0.9,
                              repetition_penalty=1.05, pad_token_id=_tok.eos_token_id)
    text = _tok.decode(out[0][tok_in.shape[-1]:], skip_special_tokens=True).strip()
    return text, retrieved, f"{time.time()-t0:.2f}s"

with gr.Blocks(title="Qwen2.5-3B + QLoRA + RAG") as demo:
    gr.Markdown("# Qwen2.5-3B (QLoRA) — Optional RAG")
    status = gr.Markdown(load())

    with gr.Row():
        system = gr.Textbox(value="You are a concise, friendly assistant.", label="System")
        use_rag = gr.Checkbox(value=False, label="Use RAG (local KB)")
        top_k = gr.Slider(1,5,step=1,value=3,label="Top-K")

    user = gr.Textbox(lines=4, label="User")
    btn = gr.Button("Ask")
    out = gr.Textbox(lines=12, label="Assistant")
    retrieved = gr.Textbox(label="Retrieved Titles")
    latency = gr.Textbox(label="Latency (s)")

    btn.click(chat, inputs=[system, user, use_rag, top_k], outputs=[out, retrieved, latency])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)