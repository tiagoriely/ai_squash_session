# src/grammar_tools/grammar_generator.py
"""
Generate synthetic squash-training sessions that respect your YAML grammar.

Usage
-----
python -m grammar_tools.grammar_generator \
       --num 100 \
       --minutes 60 \
       --outfile data/eval_dataset.jsonl
"""
from __future__ import annotations
import argparse, json, random, textwrap
from pathlib import Path
from datetime import timedelta

from .planner import build_session, WARMUP_MIN, REST_MIN, POINTS_PER_CG
from .loader  import load_yaml_dir, Activity

SEPS = "-" * 79


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _fmt_duration(mins: float) -> str:
    """Return 'X min Y s' with no trailing zero seconds."""
    td = timedelta(minutes=mins)
    mm, ss = divmod(int(td.total_seconds()), 60)
    return f"{mm} min" + (f" {ss}s" if ss else "")


def _render_warmup(act: Activity) -> list[str]:
    dur = _fmt_duration(WARMUP_MIN)
    return [f"Warm-up  (total {dur})",
            f"1. {dur}: {act.name}"]


def _render_session_blocks(blocks: list[tuple[Activity, int]]) -> list[str]:
    lines: list[str] = ["Session"]
    for idx, (act, pts) in enumerate(blocks, 1):
        lines.append(f"Condition Game {idx}")
        lines.append(f"• {pts} pts: {act.name}")
        if act.rules:                                 # short, inline rule view
            main_rule = next(iter(act.rules.values())) if isinstance(act.rules, dict) else act.rules
            lines[-1] += f" ({main_rule})"
        lines.append(f"Rest {_fmt_duration(REST_MIN)}")
        lines.append("")                              # blank between games
    return lines[:-2]                                 # drop last blank + rest


def build_one_session(total_min: int = 60) -> dict[str, str]:
    """Return dict with 'text' and metadata."""
    warmup, blocks = build_session(total_min)
    text_lines = [
        f"Duration: {total_min} min",
        "Session Focus: <auto-fill later>",
        SEPS,
        *_render_warmup(warmup),
        SEPS,
        *_render_session_blocks(blocks),
        "",
        "End of session."
    ]
    return {
        "length_min": total_min,
        "warmup_id": warmup.id,
        "block_ids": [b.id for b, _ in blocks],
        "text": "\n".join(text_lines)
    }


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic squash sessions")
    p.add_argument("-n", "--num", type=int, default=50,
                   help="number of sessions to create")
    p.add_argument("-m", "--minutes", type=int, choices=[45, 60, 90],
                   default=60, help="session length")
    p.add_argument("-o", "--outfile", type=Path,
                   default=Path("data/eval_dataset.jsonl"),
                   help="where to write the JSON-Lines dataset")
    p.add_argument("--seed", type=int, default=42, help="rng seed for repeatability")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as fh:
        for _ in range(args.num):
            record = build_one_session(args.minutes)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅  Wrote {args.num} sessions to {args.outfile}")


if __name__ == "__main__":
    main()
