"""Summarize ``outputs/eval_compare.jsonl`` into ``outputs/EVAL_SUMMARY.json``.

Upgrades over legacy implementation:
- Reports both word-level (``avg_len_base``, ``avg_len_adapter``, ``avg_words_base``,
  ``avg_words_adapter``) and character-level (``avg_chars_base``, ``avg_chars_adapter``,
  ``avg_chars_gold``) length statistics, resolving the words-vs-chars documentation
  ambiguity.
- Computes deterministic ROUGE-L (LCS F1) scores across Base, Adapter, and optional
  RAG columns without requiring external network downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "outputs" / "eval_compare.jsonl"
OUT = PROJECT_ROOT / "outputs" / "EVAL_SUMMARY.json"


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute Longest Common Subsequence (LCS) token length."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = temp
    return dp[-1]


def compute_rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 score in ``[0.0, 1.0]``."""
    pred_tokens = re.findall(r"[a-z0-9]+", (prediction or "").lower())
    ref_tokens = re.findall(r"[a-z0-9]+", (reference or "").lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    return 2.0 * prec * rec / (prec + rec)


def summarize_eval_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation rows into a structured metrics dictionary."""
    n = len(rows)
    if n == 0:
        return {
            "num_rows": 0,
            "avg_len_base": 0.0,
            "avg_len_adapter": 0.0,
            "avg_words_base": 0.0,
            "avg_words_adapter": 0.0,
            "avg_words_gold": 0.0,
            "avg_chars_base": 0.0,
            "avg_chars_adapter": 0.0,
            "avg_chars_gold": 0.0,
            "conciseness_reduction_pct": 0.0,
            "rougeL_base": 0.0,
            "rougeL_adapter": 0.0,
            "has_canberra_case": False,
            "example_canberra": None,
        }

    words_base = [len(str(r.get("base", "")).split()) for r in rows]
    words_adap = [len(str(r.get("adapter", "")).split()) for r in rows]
    words_gold = [len(str(r.get("gold", "")).split()) for r in rows]

    chars_base = [len(str(r.get("base", ""))) for r in rows]
    chars_adap = [len(str(r.get("adapter", ""))) for r in rows]
    chars_gold = [len(str(r.get("gold", ""))) for r in rows]

    avg_len_base = statistics.mean(words_base)
    avg_len_adap = statistics.mean(words_adap)
    avg_words_gold = statistics.mean(words_gold)

    avg_chars_base = statistics.mean(chars_base)
    avg_chars_adap = statistics.mean(chars_adap)
    avg_chars_gold = statistics.mean(chars_gold)

    reduction_pct = (
        round(100.0 * (1.0 - (avg_len_adap / avg_len_base)), 2)
        if avg_len_base > 0
        else 0.0
    )

    rouge_base = round(
        statistics.mean(compute_rouge_l_f1(str(r.get("base", "")), str(r.get("gold", ""))) for r in rows),
        4,
    )
    rouge_adap = round(
        statistics.mean(compute_rouge_l_f1(str(r.get("adapter", "")), str(r.get("gold", ""))) for r in rows),
        4,
    )

    canberra = next(
        (r for r in rows if "capital of australia" in str(r.get("user", "")).lower()),
        None,
    )

    summary: Dict[str, Any] = {
        "num_rows": n,
        "avg_len_base": round(avg_len_base, 4),
        "avg_len_adapter": round(avg_len_adap, 4),
        "avg_words_base": round(avg_len_base, 4),
        "avg_words_adapter": round(avg_len_adap, 4),
        "avg_words_gold": round(avg_words_gold, 4),
        "avg_chars_base": round(avg_chars_base, 2),
        "avg_chars_adapter": round(avg_chars_adap, 2),
        "avg_chars_gold": round(avg_chars_gold, 2),
        "conciseness_reduction_pct": reduction_pct,
        "rougeL_base": rouge_base,
        "rougeL_adapter": rouge_adap,
        "has_canberra_case": bool(canberra),
        "example_canberra": canberra,
    }

    if any("adapter_rag" in r for r in rows):
        summary["rougeL_adapter_rag"] = round(
            statistics.mean(
                compute_rouge_l_f1(str(r.get("adapter_rag", "")), str(r.get("gold", "")))
                for r in rows
            ),
            4,
        )
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize eval_compare.jsonl into EVAL_SUMMARY.json")
    ap.add_argument("--inp", default=str(INP))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    inp_path = Path(args.inp)
    out_path = Path(args.out)

    rows: List[Dict[str, Any]] = []
    with open(inp_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    summary = summarize_eval_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()