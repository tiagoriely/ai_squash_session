# src/grammar_tools/grammar_generator_dedup.py
"""
Generate synthetic squash-training sessions by orchestrating the planner and renderer.
This is the final, corrected version that is fully compatible with the entire grammar.
"""
from __future__ import annotations

import argparse
import copy
import json
import csv
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional
from itertools import groupby
from collections import Counter
import yaml

# DSL
from src.grammar_tools.dsl_tools.structure_validator import validate_session_structure





# Planner
from src.grammar_tools.engine.planner import Planner
# Advanced loader for exercises
from src.grammar_tools.engine.loader import load_yaml_dir, Activity

# Dedup helpers
from src.grammar_tools.dedup.hash_dedup import HashDeduper
from src.grammar_tools.dedup.jaccard_dedup import JaccardDeduper
from src.grammar_tools.dedup.exhaustion import ExhaustionStopper

# Semantic near-dedup (embedding-based)
try:
    from src.grammar_tools.dedup.semantic_dedup import (
        SemanticDeduper,
        SemanticDedupNotAvailable,
    )
except Exception:  # pragma: no cover
    SemanticDeduper = None  # type: ignore
    SemanticDedupNotAvailable = RuntimeError  # type: ignore

# --- Constants ---
SEPS = "-" * 79

# --- Config helpers ---
def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Config YAML must define a mapping at the top level.")
        return {str(k).replace("-", "_"): v for k, v in cfg.items()}


def merge_config_with_args(parser: argparse.ArgumentParser, args: argparse.Namespace,
                           cfg: Dict[str, Any]) -> argparse.Namespace:
    defaults_ns = parser.parse_args([])
    args_d, defaults_d = vars(args).copy(), vars(defaults_ns)
    for k, v in cfg.items():
        if k in args_d and args_d[k] == defaults_d.get(k):
            args_d[k] = v
    return argparse.Namespace(**args_d)


def _coerce_arg_types(args: argparse.Namespace) -> argparse.Namespace:
    for key in ("outfile", "config"):
        val = getattr(args, key, None)
        if val and not isinstance(val, Path):
            setattr(args, key, Path(val).expanduser())
    return args


# --- Loading Helper ---
def load_grammar(grammar_path: Path) -> Dict[str, Any]:
    """
    Loads all grammar components, using the new loader for all activities
    and correctly structures the data for the planner.
    """
    print("Loading and validating grammar...")

    all_activities = load_yaml_dir(root=grammar_path / "exercises")

    warmups = [v.model_dump() for v in all_activities.values() if "warmup" in v.id.lower()]
    exercise_activities = {k: v for k, v in all_activities.items() if "warmup" not in v.id.lower()}

    exercises_by_family = {}

    def get_family_id(act: Activity) -> str:
        return act.family_id or ""

    sorted_activities = sorted(exercise_activities.values(), key=get_family_id)

    for family_id, variant_group in groupby(sorted_activities, key=get_family_id):
        if not family_id: continue
        variants = [v.model_dump() for v in variant_group]
        if not variants: continue

        exercises_by_family[family_id] = {
            "family": variants[0].get('family'),
            "family_id": family_id,
            "category": variants[0].get('category'),
            "parameters": variants[0].get('parameters'),
            "variants": variants
        }

    print(f"  -> Loaded and validated {len(exercises_by_family)} exercise families and {len(warmups)} warmups.")

    def _load_simple_yaml_files(directory: Path) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for path in directory.glob("*.yaml"):
            with open(path, "r", encoding="utf-8") as f:
                data[path.stem] = yaml.safe_load(f)
        return data

    def _load_nested_yaml_files(directory: Path) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        # Iterate through subdirectories (drills, mix, etc.)
        for subdir in directory.iterdir():
            if subdir.is_dir():
                category_name = subdir.name
                data[category_name] = {}
                for path in subdir.glob("*.yaml"):
                    with open(path, "r", encoding="utf-8") as f:
                        data[category_name][path.stem] = yaml.safe_load(f)
        return data

    session_types_path = grammar_path.parent / "sessions_types.yaml"
    if not session_types_path.is_file():
        raise FileNotFoundError(f"Could not find 'sessions_types.yaml' at {session_types_path}")

    with session_types_path.open("r", encoding="utf-8") as f:
        session_types = yaml.safe_load(f)

    block_types_dict = _load_simple_yaml_files(grammar_path / "session_block_types")
    block_types = [v for v in block_types_dict.values() if isinstance(v, dict)]

    return {
        "exercises": exercises_by_family,
        "structures": _load_nested_yaml_files(grammar_path / "session_structures"),
        "session_types": session_types,
        "warmups": warmups,
        "block_types": block_types,
        "archetypes": _load_simple_yaml_files(grammar_path / "session_archetypes"),
    }


# --- Rendering Helper ---
def render_session_to_text(session_plan: Dict[str, Any]) -> str:
    meta = session_plan["meta"]
    focus = meta.get('family_focus') or meta.get('focus') or meta.get('archetype', 'General')
    lines = [
        f"Duration: {meta['duration']} min",
        f"Session Focus: {focus} (Archetype: {meta['archetype']})",
        SEPS,
    ]

    for block in session_plan["blocks"]:
        lines.append(f"### {block['name']} ###")
        # The planner now returns a 4-item tuple, so we unpack all four.
        # The last item (estimated_duration) is ignored with '_' as it's not needed for rendering.
        for exercise, value, mode, _ in block["exercises"]:
            name = exercise.get("name", "Unnamed Exercise")
            side_list = exercise.get("shotSide")
            full_name = (
                f"{name} ({', '.join(s.capitalize() for s in side_list)})"
                if side_list
                else name
            )

            if mode == "timed":
                goal_str = f"{value} min"
                header = f"Drill: {full_name}"
            else:
                goal_str = f"{value} pts"
                header = f"Conditioned Game: {full_name}"

            lines.append(f"• {goal_str}: {header}")

            rules = exercise.get("rules", {}) or {}
            constraint = rules.get("constraint") or rules.get("constraint_text")

            # Check if the constraint is a list and join it into a single string
            if isinstance(constraint, list):
                constraint = " ".join(constraint)

            if constraint:
                short_rule = textwrap.shorten(constraint, width=70, placeholder="...")
                lines.append(f"  (Rule: {short_rule})")

        if "Activity" in block["name"] and block.get("exercises"):
            lines.append(f"• Rest: {meta['rest_minutes']} min")
        lines.append("")

    lines.append(SEPS)
    lines.append("End of session.")
    return "\n".join(lines)

def jaccard_view(text: str) -> str:
    """
    Keep only exercise bullet lines and drop boilerplate (headers, separators, 'Rest' lines).
    Your renderer uses '• ' bullets and '• Rest:' for rest.
    """
    keep = []
    for ln in text.splitlines():
        if ln.startswith("• ") and not ln.startswith("• Rest:"):
            keep.append(ln.strip())
    return "\n".join(keep)


# --- CLI / Main ---
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic squash sessions from a grammar"
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of sessions")
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=Path("data/processed/squash_dataset.jsonl"),
        help="Output JSON-Lines file",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--dedup-by",
        dest="dedup_by",
        choices=["plan", "text", "none"],
        default="plan",
        help="Deduplicate by hashing the canonical 'plan' JSON, the rendered 'text', or disable",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        dest="max_attempts_multiplier",
        type=float,
        default=10.0,
        help="Stop after num * this multiplier attempts if we keep hitting duplicates.",
    )
    parser.add_argument(
        "--consecutive-dup-limit",
        type=int,
        default=0,
        help="Optional early stop after this many consecutive duplicates (0 disables).",
    )
    parser.add_argument(
        "--jaccard-dedup", action="store_true", help="Enable Jaccard near-dup filter."
    )
    parser.add_argument("--jaccard-ngram", type=int, default=2, help="n for n-grams.")
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.90,
        help="Jaccard similarity threshold (>= means near-duplicate).",
    )
    parser.add_argument(
        "--jaccard-window",
        type=int,
        default=500,
        help="Compare against the last N accepted samples (0 = all).",
    )
    parser.add_argument(
        "--semantic-dedup",
        action="store_true",
        help="Enable semantic near-duplicate filtering via sentence embeddings.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.93,
        help="Cosine similarity threshold (>= means near-duplicate). Typical: 0.90–0.97.",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name for semantic dedup.",
    )
    parser.add_argument(
        "--semantic-device",
        type=str,
        default=None,
        help="Device for Sentence-Transformers (e.g., 'cpu', 'cuda'). Defaults to auto.",
    )

    parser.add_argument(
        "--grammar-profile",
        type=str,
        default=None,  # Default to None to ensure it's set in the config
        help="The name of the grammar profile to use (e.g., 'balanced_grammar').",
    )

    parser.add_argument(
        "--json-indent",
        type=int,
        default=None,
        help="Indent the output JSONL for readability. Provide a number for indent level (e.g., 4).",
    )

    parser.add_argument(
        "--ebnf-file",
        type=str,
        default=None,
        help="Filename of the EBNF structure to use for validation.",
    )
    parser.add_argument(
        "--planner-config",
        type=dict,
        default={},
        help="Dictionary of settings to pass to the Planner.",
    )

    return parser


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config(getattr(args, "config", None))
    if cfg:
        args = merge_config_with_args(parser, args, cfg)
    args = _coerce_arg_types(args)
    return args



# ---- main -----


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    mult = args.max_attempts_multiplier
    if not (mult > 0 and mult < 1e6):
        print("⚠️  Invalid --max-attempts-multiplier; falling back to 10.0")
        mult = 10.0

    # Initialise counters early so log_skip can see them
    count = 0
    attempts = 0
    max_attempts = max(args.num, int(args.num * mult))

    # --- skip logging (lives inside main; uses args/attempts) ---
    skip_log_path = args.outfile.with_suffix(".skips.csv")
    accepted_ids: list[str] = []

    def log_skip(reason: str, score: float = -1.0, match_idx: int = -1):
        match_id = accepted_ids[match_idx] if (0 <= match_idx < len(accepted_ids)) else ""
        header = ["attempt", "reason", "score", "match_idx", "match_session_id"]
        row = [attempts, reason, f"{score:.4f}" if score >= 0 else "", match_idx, match_id]
        write_header = not skip_log_path.exists()
        with open(skip_log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
    # ------

    # Check that the grammar_profile is set in the config.
    if not args.grammar_profile:
        raise ValueError("Your config file must specify a 'grammar_profile' (e.g., 'balanced_grammar').")

    # Build the dynamic grammar path.
    grammar_path = Path("grammar/sports/squash") / args.grammar_profile
    if not grammar_path.is_dir():
        raise FileNotFoundError(f"Grammar profile directory not found at: {grammar_path}")

    print(f"Loading grammar from profile: '{args.grammar_profile}' at path: {grammar_path}")
    grammar = load_grammar(grammar_path)

    # Load RAG config
    FIELD_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval/squash_field_retrieval_config.yaml")
    if not FIELD_RETRIEVAL_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"RAG config not found at: {FIELD_RETRIEVAL_CONFIG_PATH}")
    with FIELD_RETRIEVAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        field_retrieval_data = yaml.safe_load(f)

    planner = Planner(
        exercises=grammar["exercises"],
        structures=grammar["structures"],
        session_types=grammar["session_types"],
        warmups=grammar["warmups"],
        block_types=grammar["block_types"],
        archetypes=grammar["archetypes"],
        field_retrieval_config=field_retrieval_data,
        config=args.planner_config
    )

    hash_deduper = None if args.dedup_by == "none" else HashDeduper(mode=args.dedup_by)
    jaccard_deduper: Optional[JaccardDeduper] = None
    if args.jaccard_dedup:
        jaccard_deduper = JaccardDeduper(
            ngram=args.jaccard_ngram,
            threshold=args.jaccard_threshold,
            window=args.jaccard_window,
        )
    semantic_deduper: Optional[SemanticDeduper] = None
    if args.semantic_dedup:
        try:
            assert SemanticDeduper is not None
            semantic_deduper = SemanticDeduper(
                model_name=args.semantic_model,
                device=args.semantic_device,
                threshold=args.semantic_threshold,
            )
        except Exception as e:
            print(f"⚠️  Semantic dedup disabled: {e}\n (Install deps: pip install sentence-transformers numpy)")

    stopper = ExhaustionStopper(limit=args.consecutive_dup_limit) if args.consecutive_dup_limit > 0 else None

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    near_dup_skips_sem, near_dup_skips_jac, exact_dup_skips = 0, 0, 0

    with args.outfile.open("w", encoding="utf-8") as fh:
        while count < args.num and attempts < max_attempts:
            attempts += 1
            session_plan = planner.plan_session()
            if not session_plan:
                log_skip("no_plan", -1.0, -1)
                if stopper:
                    stopper.register_dup()
                    if stopper.exhausted:
                        print(f"🛑 Stopping early: {stopper.consec} consecutive rejects/dups.")
                        break
                continue


            # Get the EBNF filename directly from the run config args
            if args.ebnf_file:
                ebnf_path = grammar_path / args.ebnf_file
                if not validate_session_structure(session_plan, ebnf_path):
                    log_skip("invalid_structure", -1.0, -1)
                    if stopper:
                        stopper.register_dup()
                        if stopper.exhausted:
                            break
                    continue

            text_output = render_session_to_text(session_plan)
            is_dup, session_hash = False, ""
            if hash_deduper is not None:
                is_dup, session_hash = hash_deduper.is_duplicate(plan=session_plan, text=text_output)
            if is_dup:
                exact_dup_skips += 1
                log_skip("exact", 1.0, -1)
                if stopper:
                    stopper.register_dup()
                    if stopper.exhausted:
                        print(f"🛑 Stopping early: {stopper.consec} consecutive rejects/dups.")
                        break
                continue

            if jaccard_deduper is not None:
                is_dup_j, sim_j, idx_j = jaccard_deduper.is_near_duplicate(jaccard_view(text_output))
                if is_dup_j:
                    near_dup_skips_jac += 1
                    log_skip("jaccard", sim_j, idx_j)
                    if stopper:
                        stopper.register_dup()
                        if stopper.exhausted:
                            print(f"🛑 Stopping early: {stopper.consec} consecutive rejects/dups.")
                            break
                    continue

            if semantic_deduper is not None:
                is_dup_s, sim_s, idx_s = semantic_deduper.is_near_duplicate(text_output)
                if is_dup_s:
                    near_dup_skips_sem += 1
                    log_skip("semantic", sim_s, idx_s)
                    if stopper:
                        stopper.register_dup()
                        if stopper.exhausted:
                            print(f"🛑 Stopping early: {stopper.consec} consecutive rejects/dups.")
                            break
                    continue

            # Determine the indent level from your config (e.g., 4 or None)
            indent_level = args.json_indent if args.json_indent and args.json_indent > 0 else None

            accepted_id = f"session_{count + 1:03d}"
            record = {
                "session_id": accepted_id,
                "meta": copy.deepcopy(session_plan["meta"]),
                "contents": text_output,
            }
            if session_hash:
                record["hash"] = session_hash
            fh.write(json.dumps(record, ensure_ascii=False, indent=indent_level) + "\n")

            accepted_ids.append(accepted_id)

            if hash_deduper and session_hash:
                hash_deduper.mark_seen(session_hash)
            if jaccard_deduper:
                jaccard_deduper.add(jaccard_view(text_output))
            if semantic_deduper:
                semantic_deduper.add(text_output)
            count += 1
            if stopper:
                stopper.register_unique()

    parts = []
    if args.dedup_by != "none": parts.append(f"exact={args.dedup_by}")
    if args.jaccard_dedup: parts.append(f"jaccard(n={args.jaccard_ngram}, thr={args.jaccard_threshold:.2f})")
    if args.semantic_dedup and semantic_deduper: parts.append(f"semantic(thr={args.semantic_threshold:.2f})")
    mode_str = ", ".join(parts)

    print(
        f"\n✅ Wrote {count} unique sessions [{mode_str}] to {args.outfile} "
        f"(attempted {attempts}, exact-dups {exact_dup_skips}, "
        f"jaccard-dups {near_dup_skips_jac}, semantic-dups {near_dup_skips_sem})"
    )


if __name__ == "__main__":
    main()