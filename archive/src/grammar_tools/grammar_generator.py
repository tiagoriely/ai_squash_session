# src/grammar_tools/grammar_generator.py
"""
Generate synthetic squash-training sessions by orchestrating the planner and renderer.

This script loads the grammar, uses the Planner to create a structured plan,
and then renders that plan into a human-readable text format.

Usage
-----
python -m src.grammar_tools.grammar_generator \
       --num 10 \
       --outfile data/processed/squash_dataset.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import yaml
import textwrap
from pathlib import Path
from typing import Any, Dict

# Make sure the planner is imported
from src.grammar_tools.engine.planner import Planner

# --- Constants ---
SEPS = "-" * 79
# This path assumes the script is run from the project root.
GRAMMAR_PATH = Path("grammar/sports") / "squash"

# --- Loading Helper ---
def load_grammar(grammar_path: Path) -> Dict[str, Any]:
    """Loads all necessary grammar components into a dictionary."""

    def _load_yaml_files(directory: Path) -> Dict[str, Any]:
        """Loads all .yaml/.json files from a directory, keyed by file stem."""
        data = {}
        for path in directory.glob('*.yaml'):
             with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                key = path.stem
                if isinstance(content, dict) and "family_id" in content:
                    key = content["family_id"]
                data[key] = content
        return data

    # Load exercise families (excluding warmups/defaults)
    exercises = {}
    ex_path = grammar_path / "exercises"
    for path in ex_path.glob('*.yaml'):
        if path.stem in ['warmups', 'defaults']:
            continue
        with open(path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if isinstance(content, dict) and "family_id" in content:
                exercises[content["family_id"]] = content

    # Load warmups, session types, and block types
    with (ex_path / "warmups.yaml").open('r', encoding='utf-8') as f:
        warmups = yaml.safe_load(f)
    with (grammar_path / "sessions_types.yaml").open('r', encoding='utf-8') as f:
        session_types = yaml.safe_load(f)
    with (grammar_path / "session_block_types.yaml").open('r', encoding='utf-8') as f:
        block_types = yaml.safe_load(f)

    # <<< CHANGE HERE: Load the session archetypes >>>
    archetypes = _load_yaml_files(grammar_path / "session_archetypes")

    return {
        "exercises": exercises,
        "structures": _load_yaml_files(grammar_path / "session_structures"),
        "session_types": session_types,
        "warmups": warmups if isinstance(warmups, list) else warmups.get('variants', []),
        "block_types": block_types,
        "archetypes": archetypes, # <<< CHANGE HERE: Add archetypes to the returned dictionary >>>
    }

# --- Rendering Helper ---
def render_session_to_text(session_plan: Dict[str, Any]) -> str:
    """Renders a structured session plan into a human-readable string."""
    meta = session_plan['meta']
    focus = meta.get('family_focus') or meta.get('focus') or meta.get('archetype', 'General')
    lines = [
        f"Duration: {meta['duration']} min",
        f"Session Focus: {focus} (Archetype: {meta['archetype']})",
        SEPS,
    ]

    for block in session_plan["blocks"]:
        lines.append(f"### {block['name']} ###")
        for exercise, value, mode in block['exercises']:
            name = exercise.get('name', 'Unnamed Exercise')
            full_name = f"{name} ({exercise['shotSide'].capitalize()})" if exercise.get('shotSide') else name

            if mode == "timed":
                goal_str = f"{value} min"
                header = f"Drill: {full_name}"
            else:
                goal_str = f"{value} pts"
                header = f"Conditioned Game: {full_name}"

            lines.append(f"• {goal_str}: {header}")

            constraint = exercise.get("rules", {}).get("constraint", "")
            if constraint:
                short_rule = textwrap.shorten(constraint, width=70, placeholder="...")
                lines.append(f"  (Rule: {short_rule})")

        if "Activity" in block['name']:
            lines.append(f"• Rest: {meta['rest_minutes']} min")
        lines.append("")

    lines.append(SEPS)
    lines.append("End of session.")
    return "\n".join(lines)


# --- Main Execution ---
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic squash sessions from a grammar")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of sessions to create")
    parser.add_argument("-o", "--outfile", type=Path, default=Path("data/processed/squash_dataset.jsonl"), help="Output JSON-Lines file")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for repeatability")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    if args.seed:
        random.seed(args.seed)

    # 1. Load the entire grammar once, including archetypes
    grammar = load_grammar(GRAMMAR_PATH)

    # 2. Instantiate the planner with all required grammar components
    planner = Planner(
        exercises=grammar['exercises'],
        structures=grammar['structures'],
        session_types=grammar['session_types'],
        warmups=grammar['warmups'],
        block_types=grammar['block_types'],
        archetypes=grammar['archetypes'] # <<< CHANGE HERE: Pass the loaded archetypes to the Planner >>>
    )

    # 3. Generate and write sessions
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as fh:
        count = 0
        for i in range(args.num):
            session_plan = planner.plan_session()

            if session_plan:
                text_output = render_session_to_text(session_plan)
                record = {
                    "session_id": f"session_{i + 1:03d}",
                    "meta": session_plan["meta"],
                    "text": text_output
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"\n✅ Successfully wrote {count} sessions to {args.outfile}")

if __name__ == "__main__":
    main()