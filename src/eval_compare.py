"""Evaluate Base vs. QLoRA Adapter (and optional Confidence-Gated RAG) on validation samples.

Upgrades over legacy implementation:
- Loads the 4-bit base model + QLoRA adapter ONCE and toggles ``with model.disable_adapter():``
  for base-model generation, cutting evaluation VRAM in half (~2.5 GB vs ~4.9 GB).
- Supports ``--include_rag`` and ``--score_threshold`` for a 4-way ablation study:
  1. Base (no RAG)
  2. Base + Confidence-Gated RAG
  3. QLoRA Adapter (no RAG)
  4. QLoRA Adapter + Confidence-Gated RAG
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_KB_DIR,
    build_bnb4_config,
    device_map_single,
    generate_from_messages,
    generate_response,
    load_qwen_model,
)
from rag_utils import build_rag_prompt, init_kb  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Single-model Base vs QLoRA Adapter (+RAG) evaluator")
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER_DIR))
    ap.add_argument("--val_path", default=str(PROJECT_ROOT / "data" / "processed" / "val.jsonl"))
    ap.add_argument("--out_path", default=str(PROJECT_ROOT / "outputs" / "eval_compare.jsonl"))
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--include_rag", action="store_true", help="Also evaluate Base+RAG and Adapter+RAG")
    ap.add_argument("--kb_dir", default=str(DEFAULT_KB_DIR))
    ap.add_argument("--score_threshold", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def bnb4():
    """Return 4-bit NF4 quantization configuration."""
    return build_bnb4_config()


def load_adapter(base_id: str, adapter_dir: str):
    """Load 4-bit base model with attached PeftModel adapter."""
    return load_qwen_model(base_id=base_id, adapter_dir=adapter_dir, load_adapter=True)


def gen_text(tok: Any, model: Any, messages: List[Dict[str, str]], max_new_tokens: int, disable_adapter: bool = False) -> str:
    """Generate response from ``messages``, optionally disabling LoRA adapter weights."""
    return generate_from_messages(
        tok,
        model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05,
        disable_adapter=disable_adapter,
    )


def main(argv=None):
    from datasets import load_dataset
    import evaluate

    args = parse_args(argv)
    ds = load_dataset("json", data_files={"val": args.val_path})["val"]
    idxs = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idxs)
    idxs = idxs[: args.num_samples]

    # Load ONCE; base predictions use `disable_adapter=True`
    tok, model = load_adapter(args.base, args.adapter)
    if args.include_rag:
        init_kb(args.kb_dir)

    rouge = evaluate.load("rouge")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, k in enumerate(idxs, 1):
        msgs = ds[k]["messages"]
        gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        user_msgs = [m for m in msgs if m["role"] != "assistant"]
        user_query = user_msgs[-1]["content"] if user_msgs else ""
        sys_prompt = next(
            (m["content"] for m in user_msgs if m["role"] == "system"),
            "You are a concise, friendly assistant.",
        )

        tb0 = time.time()
        pred_base = gen_text(tok, model, user_msgs, args.max_new_tokens, disable_adapter=True)
        base_lat = round(time.time() - tb0, 3)

        ta0 = time.time()
        pred_adap = gen_text(tok, model, user_msgs, args.max_new_tokens, disable_adapter=False)
        adap_lat = round(time.time() - ta0, 3)

        row: Dict[str, Any] = {
            "idx": k,
            "user": user_query,
            "gold": gold,
            "base": pred_base,
            "adapter": pred_adap,
            "base_latency_s": base_lat,
            "adapter_latency_s": adap_lat,
        }

        if args.include_rag:
            rag_plan = build_rag_prompt(
                system_prompt=sys_prompt,
                user_query=user_query,
                use_rag=True,
                top_k=3,
                score_threshold=args.score_threshold,
                strict_kb=False,
                kb_dir=args.kb_dir,
            )
            row["rag_mode"] = rag_plan["mode"]
            row["base_rag"] = generate_response(
                tok,
                model,
                system=rag_plan["system"],
                user=rag_plan["user"],
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                disable_adapter=True,
            )
            row["adapter_rag"] = generate_response(
                tok,
                model,
                system=rag_plan["system"],
                user=rag_plan["user"],
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                disable_adapter=False,
            )

        rows.append(row)
        if i % 10 == 0:
            print(f"Processed {i}/{len(idxs)}")

    base_scores = rouge.compute(
        predictions=[r["base"] for r in rows],
        references=[r["gold"] for r in rows],
    )
    adap_scores = rouge.compute(
        predictions=[r["adapter"] for r in rows],
        references=[r["gold"] for r in rows],
    )

    dt = time.time() - t0
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== ROUGE-L F1 ===")
    print("Base   :", round(base_scores["rougeLsum"], 4))
    print("Adapter:", round(adap_scores["rougeLsum"], 4))
    if args.include_rag:
        base_rag_scores = rouge.compute(
            predictions=[r["base_rag"] for r in rows],
            references=[r["gold"] for r in rows],
        )
        adap_rag_scores = rouge.compute(
            predictions=[r["adapter_rag"] for r in rows],
            references=[r["gold"] for r in rows],
        )
        print("Base+RAG   :", round(base_rag_scores["rougeLsum"], 4))
        print("Adapter+RAG:", round(adap_rag_scores["rougeLsum"], 4))
    print(f"Wall time: {dt / 60:.1f} min for {len(rows)} samples")
    print(f"Saved details to {out_path}")


if __name__ == "__main__":
    main()