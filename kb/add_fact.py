"""Append a verified fact record to ``kb/facts.jsonl``.

Fixes over legacy implementation:
- Root-anchors ``FACTS = Path(__file__).resolve().parent / "facts.jsonl"`` so it
  works from any working directory.
- Replaces deprecated ``datetime.utcnow()`` with ``datetime.now(timezone.utc)``.
- Prevents duplicate fact IDs and validates non-empty title/text fields.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Union

FACTS = Path(__file__).resolve().parent / "facts.jsonl"


def append_fact(
    title: str,
    text: str,
    fact_id: str | None = None,
    facts_path: Union[str, Path] = FACTS,
) -> Dict[str, Any]:
    """Validate and append a single fact record to ``facts_path``."""
    clean_title = (title or "").strip()
    clean_text = (text or "").strip()
    if not clean_title or not clean_text:
        raise ValueError("Both 'title' and 'text' must be non-empty strings.")

    target = Path(facts_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    resolved_id = (
        fact_id.strip()
        if fact_id and fact_id.strip()
        else f"fact_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )

    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing = json.loads(line)
                if existing.get("id") == resolved_id:
                    raise ValueError(f"Duplicate fact ID '{resolved_id}' already exists in {target}")

    record = {"id": resolved_id, "title": clean_title, "text": clean_text}
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Append a fact to kb/facts.jsonl")
    ap.add_argument("--title", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--facts_path", default=str(FACTS))
    args = ap.parse_args(argv)

    rec = append_fact(
        title=args.title,
        text=args.text,
        fact_id=args.id,
        facts_path=args.facts_path,
    )
    print(f"Appended {rec['id']}. Now rebuild index:\n  python src/rag_build.py")


if __name__ == "__main__":
    main()