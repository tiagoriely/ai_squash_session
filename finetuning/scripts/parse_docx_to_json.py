#!/usr/bin/env python
"""
docx_to_json.py
================
Convert every DOCX under **data/raw/** into a single JSON‑lines file that follows
`finetuning/data/template_schema.json`.

Each JSON object represents **one complete session**.  Exercises that contain
progressive mini‑sets are stored with a nested `segments` array so no
timing/detail is lost.

Run from repo root (with venv activated):

```bash
python -m finetuning.scripts.docx_to_json
```

The output is written to
`finetuning/data/finetune_splits/train.jsonl` and will be overwritten each run.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import timedelta
from pathlib import Path

from docx import Document

# ---------------------------------------------------------------------------
# Paths & regexes
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # repo root
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = (
    ROOT / "finetuning" / "data" / "finetune_splits" / "train.jsonl"
)

# Meta lines are "Key: value" pairs that appear only once.
META_RE = re.compile(
    r"^(Type|Participants|Duration|squashLevel|Intensity|Fitness|Focus|Rest time|Support doc)\s*:.*",
    re.I,
)

WARMUP_RE = re.compile(r"^warm[- ]?up", re.I)
EX_RE = re.compile(r"^Exercise\s+(\d+)", re.I)
SEG_RE = re.compile(r"^(\d+)\s*min.*?:\s*(.*)", re.I)
DUR_BRACKET = re.compile(r"\((\d+)\s*min", re.I)

# Instructions may appear as a header *with or without a colon*.
INSTR_HEAD_RE = re.compile(r"^instructions\s*:?\s*$", re.I)
# Any of these headers ends the free‑text instructions block.
END_OF_SECTION_RE = re.compile(r"^(warm[- ]?up|exercise\s+\d+)", re.I)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mm_to_sec(mm: int) -> int:
    """Return *minutes* as seconds (int)."""
    return int(timedelta(minutes=mm).total_seconds())


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_docx(path: Path) -> dict:
    """Convert one DOCX file to a session‑JSON dict."""

    doc = Document(path)

    meta: dict[str, str] = {}
    warm: list[dict] = []
    exercises: list[dict] = []
    instr_buf: list[str] = []

    curr_ex: dict | None = None
    section: str | None = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 0️⃣ Stand‑alone "Instructions" header (no colon)
        if INSTR_HEAD_RE.match(text):
            section = "instructions"
            continue

        # 1️⃣ One‑liner metadata (Key: value)
        if META_RE.match(text):
            key, val = text.split(":", 1)
            meta[key.strip().lower()] = val.strip()
            continue

        # 2️⃣ Inside free‑text instructions
        if section == "instructions":
            if END_OF_SECTION_RE.match(text):
                section = None  # instructions ended; fall through to header logic
            else:
                instr_buf.append(text)
                continue  # keep buffering lines

        # 3️⃣ Warm‑up section header
        if WARMUP_RE.match(text):
            section = "warmup"
            continue

        # 4️⃣ Exercise block header
        m_ex = EX_RE.match(text)
        if m_ex:
            # Close previous exercise if any
            if curr_ex:
                exercises.append(curr_ex)

            duration_match = DUR_BRACKET.search(text)
            curr_ex = {
                "name": f"Exercise {m_ex.group(1)}",
                "block_time_sec": mm_to_sec(int(duration_match.group(1))) if duration_match else None,
                "segments": [],
            }
            section = "exercise"
            continue

        # 5️⃣ Segment rows
        if section == "warmup":
            m_seg = SEG_RE.match(text)
            if m_seg:
                warm.append(
                    {
                        "time_sec": mm_to_sec(int(m_seg.group(1))),
                        "pattern": m_seg.group(2),
                    }
                )
        elif section == "exercise" and curr_ex:
            m_seg = SEG_RE.match(text)
            if m_seg:
                curr_ex["segments"].append(
                    {
                        "time_sec": mm_to_sec(int(m_seg.group(1))),
                        "pattern": m_seg.group(2),
                    }
                )

    # Append the final exercise
    if curr_ex:
        exercises.append(curr_ex)

    # ---------------------------------------------------------------------
    # Build the final JSON record
    # ---------------------------------------------------------------------
    record = {
        "id": uuid.uuid4().int % 1_000_000,
        "title": path.stem,
        "type": meta.get("type", "session").lower(),
        "participants": int(meta.get("participants", 2)),
        "duration_min": int(meta.get("duration", "60").replace("min", "")),
        "squash_level": meta.get("squashlevel", "Intermediate"),
        "fitness": meta.get("fitness", "Medium"),
        "intensity": meta.get("intensity", "Medium"),
        "focus": meta.get("focus", ""),
        "rest_between_blocks_sec": mm_to_sec(
            int(meta.get("rest time", "1").replace("min", ""))
        ),
        "warm_up": warm,
        "exercises": exercises,
        "optional_game": None,
        "support_doc": meta.get("support doc", path.name),
        "instructions": meta.get("instructions") or " ".join(instr_buf),
    }
    return record


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for docx in sorted(RAW_DIR.glob("*.docx")):
            record = parse_docx(docx)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
