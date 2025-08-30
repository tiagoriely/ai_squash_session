#!/usr/bin/env python

# makes deterministic splits for training

"""
split_pairs.py
--------------
Deterministically shuffles `pairs.jsonl` and writes
train/valid/test splits (80/10/10) into the same folder.

Run:
    python -m finetuning.scripts.split_pairs
"""
import hashlib, json, pathlib

root  = pathlib.Path(__file__).resolve().parents[2] / "finetuning" / "data" / "finetune_splits"
pairs = list(root.joinpath("pairs.jsonl").read_text().splitlines())

# stable shuffle based on SHA-1 of the line
pairs.sort(key=lambda x: hashlib.sha1(x.encode()).hexdigest())

n = len(pairs)
splits = {
    "train_instruct": pairs[: int(0.8 * n)],
    "valid_instruct": pairs[int(0.8 * n): int(0.9 * n)],
    "test_instruct":  pairs[int(0.9 * n):],
}

for name, subset in splits.items():
    (root / f"{name}.jsonl").write_text("\n".join(subset), encoding="utf-8")
    print(f"{name}: {len(subset)} examples")
