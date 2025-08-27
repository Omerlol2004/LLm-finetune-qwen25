from datasets import load_dataset
from transformers import AutoTokenizer
import json, os, random

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "databricks/databricks-dolly-15k"
OUT_DIR = "data/processed"
SEED = 42
TRAIN_SPLIT = 0.9
MAX_SAMPLES = None  # set e.g. 500 for a fast smoke test

def to_messages(example):
    # Dolly fields: instruction, context, response
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    resp = (example.get("response") or "").strip()

    if ctx:
        user = f"{instr}\n\nContext:\n{ctx}"
    else:
        user = instr if instr else "Follow the instruction."

    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": resp},
    ]
    return messages

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading dataset:", DATASET_ID)
    ds = load_dataset(DATASET_ID)

    records = []
    for ex in ds["train"]:
        msgs = to_messages(ex)
        records.append({"messages": msgs})

    if MAX_SAMPLES:
        records = records[:MAX_SAMPLES]

    random.Random(SEED).shuffle(records)
    n = len(records)
    n_train = int(n * TRAIN_SPLIT)
    train_recs = records[:n_train]
    val_recs = records[n_train:]

    # Sanity: render a few with tokenizer chat template (not saved; just to verify)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    def render(msgs):
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

    print("Total:", n, " | train:", len(train_recs), " | val:", len(val_recs))
    print("\nSAMPLE RENDERED TEXT (first train example):\n")
    print(render(train_recs[0]["messages"])[:800], "...\n")

    # Save JSONL (messages list per line)
    def save_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path = os.path.join(OUT_DIR, "val.jsonl")
    save_jsonl(train_path, train_recs)
    save_jsonl(val_path, val_recs)
    print("Wrote:", train_path, "and", val_path)

if __name__ == "__main__":
    main()