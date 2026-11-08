"""Confidence-Gated Retrieval-Augmented Generation (RAG) utilities.

Key upgrades over legacy implementation:
1. Root-anchored default ``KB_DIR`` so scripts work from any working directory.
2. Lazy imports of ``faiss`` and ``sentence_transformers`` inside ``init_kb()``
   with a deterministic lexical cosine fallback when running in lightweight
   CPU test environments without compiled FAISS wheels.
3. Word-boundary regex greeting detection (``is_conversational_query``) fixing
   the substring collision bug where ``"hi"`` matched ``"which"`` / ``"history"``.
4. Confidence-gated filtering (``filter_and_rank_hits``) that strips FAISS ``-1``
   padding sentinels when ``top_k > ntotal`` and enforces a minimum similarity
   threshold ``score_threshold`` (tau).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "kb"
DEFAULT_EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

CONVERSATIONAL_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you(\s+doing)?|thanks|thank\s+you|thx|bye|goodbye|see\s+ya)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)

_embedder: Any = None
_index: Any = None
_meta: Optional[Dict[str, List[str]]] = None
_kb_dir: Optional[Path] = None
_backend: str = "uninitialized"


def is_conversational_query(query: str) -> bool:
    """Return True only when the query is a standalone greeting or pleasantry.

    Uses strict word-boundary anchoring so words like 'which', 'this', 'history',
    'machine', or 'they' never trigger false positives.
    """
    if not query or not query.strip():
        return False
    return bool(CONVERSATIONAL_PATTERN.match(query))


def filter_and_rank_hits(
    scores: Sequence[Any],
    indices: Sequence[Any],
    meta: Dict[str, List[str]],
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Filter raw FAISS / similarity outputs by valid bounds and confidence threshold.

    Parameters
    ----------
    scores:
        1D or 2D sequence of inner-product / cosine similarity scores ``D``.
    indices:
        1D or 2D sequence of document indices ``I`` (may contain ``-1`` sentinels).
    meta:
        Metadata dictionary containing ``titles``, ``texts``, and optional ``ids``.
    score_threshold:
        Minimum similarity score ``tau`` required to keep a retrieved passage.
    """
    if scores is None or indices is None or not meta:
        return []

    # Unwrap 2D batch arrays (e.g., D[0], I[0] from FAISS search) if passed directly
    score_row = scores[0] if (len(scores) > 0 and isinstance(scores[0], (list, tuple)) or hasattr(scores[0], "__len__")) else scores
    index_row = indices[0] if (len(indices) > 0 and isinstance(indices[0], (list, tuple)) or hasattr(indices[0], "__len__")) else indices

    titles = meta.get("titles", [])
    texts = meta.get("texts", [])
    ids = meta.get("ids") or [f"doc_{i}" for i in range(len(titles))]
    n_docs = min(len(titles), len(texts))

    hits: List[Dict[str, Any]] = []
    threshold = float(score_threshold)

    for raw_score, raw_idx in zip(score_row, index_row):
        idx = int(raw_idx)
        if idx < 0 or idx >= n_docs:
            continue
        score = float(raw_score)
        if math.isnan(score) or score < threshold:
            continue
        hits.append(
            {
                "rank": len(hits) + 1,
                "id": ids[idx] if idx < len(ids) else f"doc_{idx}",
                "title": titles[idx],
                "text": texts[idx],
                "score": round(score, 4),
            }
        )
    return hits


def _tokenize_content(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


def _lexical_search(
    query: str,
    meta: Dict[str, List[str]],
    top_k: int = 3,
) -> tuple[List[float], List[int]]:
    """Deterministic cosine + query-coverage fallback when FAISS is unavailable."""
    titles = meta.get("titles", [])
    texts = meta.get("texts", [])
    n_docs = min(len(titles), len(texts))
    if n_docs == 0:
        return [], []

    q_tokens = _tokenize_content(query)
    if not q_tokens:
        return [], []
    q_set = set(q_tokens)

    scored: List[tuple[float, int]] = []
    for idx in range(n_docs):
        doc_tokens = _tokenize_content(f"{titles[idx]} {texts[idx]}")
        if not doc_tokens:
            scored.append((0.0, idx))
            continue
        doc_set = set(doc_tokens)
        overlap = q_set & doc_set
        if not overlap:
            scored.append((0.0, idx))
            continue
        q_coverage = len(overlap) / len(q_set)
        cosine = len(overlap) / math.sqrt(len(q_set) * len(doc_set))
        score = round(0.65 * q_coverage + 0.35 * cosine, 4)
        scored.append((score, idx))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    k = min(max(1, int(top_k)), n_docs)
    top = scored[:k]
    return [s for s, _ in top], [i for _, i in top]


def init_kb(kb_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Initialize knowledge base index and metadata from ``kb_dir``."""
    global _embedder, _index, _meta, _kb_dir, _backend

    resolved_dir = Path(kb_dir).resolve() if kb_dir is not None else KB_DIR.resolve()
    if _kb_dir == resolved_dir and _meta is not None and _backend != "uninitialized":
        return {"kb_dir": str(_kb_dir), "backend": _backend, "num_passages": len(_meta.get("titles", []))}

    index_path = resolved_dir / "index.faiss"
    meta_path = resolved_dir / "meta.json"
    if not (index_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"KB not built. Missing {index_path} or {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)
    _kb_dir = resolved_dir

    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        _index = faiss.read_index(str(index_path))
        _embedder = SentenceTransformer(DEFAULT_EMBEDDER_ID, device="cpu")
        _backend = "faiss"
    except ImportError:
        _index = None
        _embedder = None
        _backend = "lexical_fallback"

    return {"kb_dir": str(_kb_dir), "backend": _backend, "num_passages": len(_meta.get("titles", []))}


def retrieve(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.0,
    kb_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve and rank top-K passages exceeding ``score_threshold``."""
    if not query or not query.strip():
        return []

    target_dir = Path(kb_dir).resolve() if kb_dir is not None else (_kb_dir or KB_DIR.resolve())
    if _meta is None or _kb_dir != target_dir:
        init_kb(target_dir)

    assert _meta is not None
    n_docs = len(_meta.get("titles", []))
    if n_docs == 0:
        return []

    if _backend == "faiss" and _index is not None and _embedder is not None:
        if _index.ntotal == 0:
            return []
        k = min(max(1, int(top_k)), int(_index.ntotal))
        q = _embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = _index.search(q.astype("float32"), k)
        return filter_and_rank_hits(scores[0], indices[0], _meta, score_threshold=score_threshold)

    scores_1d, indices_1d = _lexical_search(query, _meta, top_k=top_k)
    return filter_and_rank_hits(scores_1d, indices_1d, _meta, score_threshold=score_threshold)


def context_block(hits: Sequence[Dict[str, Any]], include_scores: bool = False) -> str:
    """Format retrieved passage dictionaries into a numbered context block."""
    if not hits:
        return ""
    if include_scores:
        return "\n\n".join(
            [
                f"[{h['rank']}] ({h.get('id', 'kb')} | score={float(h.get('score', 0.0)):.2f}) "
                f"{h['title']}: {h['text']}"
                for h in hits
            ]
        )
    return "\n\n".join([f"[{h['rank']}] {h['title']}: {h['text']}" for h in hits])


def build_rag_prompt(
    system_prompt: str,
    user_query: str,
    use_rag: bool = True,
    top_k: int = 3,
    score_threshold: float = 0.30,
    strict_kb: bool = False,
    kb_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Construct confidence-gated system/user prompts and metadata for RAG inference.

    Returns a dictionary with keys:
      - ``system``: effective system prompt
      - ``user``: effective user prompt (augmented with ``CONTEXT`` when grounded)
      - ``hits``: list of retrieved hit dicts passing ``score_threshold``
      - ``mode``: ``"conversational"`` | ``"rag_grounded"`` | ``"rag_abstain"`` | ``"general_fallback"``
      - ``retrieved_summary``: human-readable summary string for UI/CLI display
    """
    if not use_rag:
        return {
            "system": system_prompt,
            "user": user_query,
            "hits": [],
            "mode": "general_fallback",
            "retrieved_summary": "(RAG disabled)",
        }

    if is_conversational_query(user_query):
        return {
            "system": system_prompt,
            "user": user_query,
            "hits": [],
            "mode": "conversational",
            "retrieved_summary": "(conversational - retrieval skipped)",
        }

    hits = retrieve(
        user_query,
        top_k=top_k,
        score_threshold=score_threshold,
        kb_dir=kb_dir,
    )

    if hits:
        ctx = context_block(hits)
        if strict_kb:
            effective_system = (
                "You answer using ONLY the CONTEXT below. "
                "If the answer is not in the context, say \"I don't know\" briefly. "
                "Be factual and concise."
            )
        else:
            effective_system = (
                "You are a helpful assistant. Prioritize the provided CONTEXT to ground "
                "your answer accurately, and supplement with concise general knowledge only "
                "when needed. Be factual and concise."
            )
        effective_user = f"CONTEXT:\n{ctx}\n\nQUESTION: {user_query}"
        summary = ", ".join(
            [f"{h['title']} [{h.get('id', 'kb')}: {float(h.get('score', 0.0)):.2f}]" for h in hits]
        )
        return {
            "system": effective_system,
            "user": effective_user,
            "hits": hits,
            "mode": "rag_grounded",
            "retrieved_summary": summary,
        }

    if strict_kb:
        effective_system = (
            "No verified passages in the knowledge base matched above the confidence threshold. "
            "Briefly state that the answer is not verified in the local knowledge base."
        )
        return {
            "system": effective_system,
            "user": user_query,
            "hits": [],
            "mode": "rag_abstain",
            "retrieved_summary": f"(no hits >= {score_threshold:.2f} - strict abstain)",
        }

    return {
        "system": system_prompt,
        "user": user_query,
        "hits": [],
        "mode": "general_fallback",
        "retrieved_summary": f"(no hits >= {score_threshold:.2f} - general knowledge fallback)",
    }