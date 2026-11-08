"""Build FAISS inner-product index and metadata JSON from ``kb/facts.jsonl``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "kb"
FACTS_PATH = KB_DIR / "facts.jsonl"
INDEX_PATH = KB_DIR / "index.faiss"
META_PATH = KB_DIR / "meta.json"
DEFAULT_EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_facts(path: Union[str, Path] = FACTS_PATH) -> List[Dict[str, Any]]:
    """Read and validate JSONL fact records from ``path``."""
    items: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("id", "title", "text"):
                if key not in record or not str(record[key]).strip():
                    raise ValueError(f"Missing or empty '{key}' at {path}:{line_no}")
            rec_id = str(record["id"])
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate fact id '{rec_id}' at {path}:{line_no}")
            seen_ids.add(rec_id)
            items.append(record)
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build local FAISS index from kb/facts.jsonl")
    ap.add_argument("--kb_dir", type=str, default=str(KB_DIR))
    ap.add_argument("--embedder", type=str, default=DEFAULT_EMBEDDER_ID)
    args = ap.parse_args(argv)

    kb_dir = Path(args.kb_dir).resolve()
    facts_path = kb_dir / "facts.jsonl"
    index_path = kb_dir / "index.faiss"
    meta_path = kb_dir / "meta.json"

    kb_dir.mkdir(parents=True, exist_ok=True)
    facts = load_facts(facts_path)
    texts = [f"{x['title']}\n\n{x['text']}" for x in facts]
    ids = [x["id"] for x in facts]

    import faiss  # type: ignore
    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore

    print(f"Loading embedder (CPU): {args.embedder}")
    model = SentenceTransformer(args.embedder, device="cpu")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ids": ids,
                "titles": [x["title"] for x in facts],
                "texts": [x["text"] for x in facts],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Built FAISS index:", index_path, "| passages:", len(ids))


if __name__ == "__main__":
    main()