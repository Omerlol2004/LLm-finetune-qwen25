"""Mix targeted factual/style patch rows with general rehearsal data.

Fixes over legacy implementation:
- Adds ``--patch_repeat`` (default ``8``) and ``--sample_main`` (default ``250``)
  so the 16 patch examples are upsampled to 128 rows against 250 rehearsal rows
  (~33.9% patch density instead of the legacy 1.57% dilution where 16 rows were
  buried inside 1,000 Dolly rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_MAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
IN_PATCHES = [
    PROJECT_ROOT / "data" / "processed" / "patch_facts.jsonl",
    PROJECT_ROOT / "data" / "processed" / "patch_style.jsonl",
]
OUT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train_patch.jsonl"
DEFAULT_SAMPLE_MAIN = 250
DEFAULT_PATCH_REPEAT = 8


def read_jsonl(path: Path | str) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_mixed_dataset(
    main_rows: Sequence[Dict[str, Any]],
    patch_rows: Sequence[Dict[str, Any]],
    sample_main: int = DEFAULT_SAMPLE_MAIN,
    patch_repeat: int = DEFAULT_PATCH_REPEAT,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Combine upsampled patch rows with a random rehearsal sample from ``main_rows``."""
    rng = random.Random(seed)
    sampled_main = list(main_rows)
    rng.shuffle(sampled_main)
    sampled_main = sampled_main[: max(0, int(sample_main))]

    repeated_patches = list(patch_rows) * max(1, int(patch_repeat))
    mixed = repeated_patches + sampled_main
    rng.shuffle(mixed)
    return mixed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build upsampled micro-patch training mix")
    ap.add_argument("--in_main", default=str(IN_MAIN))
    ap.add_argument("--in_patches", nargs="+", default=[str(p) for p in IN_PATCHES])
    ap.add_argument("--out_train", default=str(OUT_TRAIN))
    ap.add_argument(
        "--sample_main",
        type=int,
        default=DEFAULT_SAMPLE_MAIN,
        help="Number of general rehearsal examples sampled from train.jsonl (default: 250)",
    )
    ap.add_argument(
        "--patch_repeat",
        type=int,
        default=DEFAULT_PATCH_REPEAT,
        help="Upsampling multiplier for patch examples to prevent gradient dilution (default: 8)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    main_rows = list(read_jsonl(args.in_main))
    patch_rows: List[Dict[str, Any]] = []
    for p in args.in_patches:
        patch_rows.extend(read_jsonl(p))

    mixed = build_mixed_dataset(
        main_rows=main_rows,
        patch_rows=patch_rows,
        sample_main=args.sample_main,
        patch_repeat=args.patch_repeat,
        seed=args.seed,
    )

    out_path = Path(args.out_train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eff_patch = len(patch_rows) * max(1, int(args.patch_repeat))
    density = (100.0 * eff_patch / len(mixed)) if mixed else 0.0
    print(
        f"train_patch.jsonl size: {len(mixed)} | unique_patch: {len(patch_rows)} "
        f"(x{args.patch_repeat}={eff_patch}, {density:.1f}%) | main_sample: {min(len(main_rows), args.sample_main)}"
    )


if __name__ == "__main__":
    main()