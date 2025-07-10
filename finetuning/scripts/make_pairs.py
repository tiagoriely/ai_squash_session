#!/usr/bin/env python
"""
make_pairs.py
=============
Read the canonical *session* JSONL created by **docx_to_json.py** and wrap each
record into a *prompt / completion* pair suitable for instruction‑tuning (LoRA,
QLoRA, full‑SFT, etc.).

The generated file – `finetuning/data/finetune_splits/pairs.jsonl` – becomes the
single source for later train/valid/test splits.

Run from the repository root:

```bash
python -m finetuning.scripts.make_pairs
```
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "finetuning" / "data" / "finetune_splits"
SESSIONS_IN = SPLIT_DIR / "train.jsonl"          # produced by docx_to_json.py
PAIRS_OUT   = SPLIT_DIR / "pairs.jsonl"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Design a {duration_min}‑minute squash {type} session for a {squash_level} "
    "player that focuses on: {focus}\n\nReturn **only** JSON in the exact "
    "format provided during fine‑tuning."
)

# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def main() -> None:
    if not SESSIONS_IN.exists():
        raise FileNotFoundError(
            f"Input file {SESSIONS_IN} not found. Run docx_to_json.py first.")

    pairs: list[str] = []

    with SESSIONS_IN.open("r", encoding="utf-8") as fin:
        for raw_line in fin:
            session = json.loads(raw_line)

            prompt = PROMPT_TEMPLATE.format(**session)
            completion = json.dumps(session, ensure_ascii=False)

            pairs.append(json.dumps({
                "prompt": prompt,
                "completion": completion
            }, ensure_ascii=False))

    PAIRS_OUT.parent.mkdir(parents=True, exist_ok=True)
    PAIRS_OUT.write_text("\n".join(pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} instruction pairs → {PAIRS_OUT}")


if __name__ == "__main__":
    main()
