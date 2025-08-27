import os, json
import faiss
from sentence_transformers import SentenceTransformer

_embedder = None
_index = None
_meta = None
_kb_dir = None

def init_kb(kb_dir="kb"):
    global _embedder, _index, _meta, _kb_dir
    if _kb_dir == kb_dir and _embedder and _index and _meta:
        return
    _kb_dir = kb_dir
    index_path = os.path.join(kb_dir, "index.faiss")
    meta_path  = os.path.join(kb_dir, "meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"KB not built. Missing {index_path} or {meta_path}")
    _index = faiss.read_index(index_path)
    with open(meta_path,"r",encoding="utf-8") as f:
        _meta = json.load(f)
    # CPU embedder; small + fast
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def retrieve(query, top_k=3):
    if _embedder is None: init_kb("kb")
    q = _embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    import numpy as np
    D, I = _index.search(q.astype("float32"), top_k)
    hits = []
    for rank, idx in enumerate(I[0]):
        hits.append({
            "rank": rank+1,
            "title": _meta["titles"][idx],
            "text":  _meta["texts"][idx],
        })
    return hits

def context_block(hits):
    return "\n\n".join([f"[{h['rank']}] {h['title']}: {h['text']}" for h in hits])