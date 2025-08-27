import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KB_DIR = "kb"
FACTS_PATH = os.path.join(KB_DIR, "facts.jsonl")
INDEX_PATH = os.path.join(KB_DIR, "index.faiss")
META_PATH = os.path.join(KB_DIR, "meta.json")

def load_facts(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def main():
    os.makedirs(KB_DIR, exist_ok=True)
    facts = load_facts(FACTS_PATH)
    texts = [f"{x['title']}\n\n{x['text']}" for x in facts]
    ids = [x["id"] for x in facts]

    print("Loading embedder (CPU): sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "titles": [x["title"] for x in facts], "texts": [x["text"] for x in facts]}, f, ensure_ascii=False, indent=2)

    print("Built FAISS index:", INDEX_PATH, "| passages:", len(ids))

if __name__ == "__main__":
    main()