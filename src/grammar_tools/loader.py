# src/grammar_tools/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ruamel.yaml import YAML
from pydantic import BaseModel, Field

# DSL parser (your wrapper that accepts str | list[str])
from .parser import parse_rules_sequence, ParseError as DSLParseError
# Semantic checks (hard/soft)
from .semantic_checks import run_semantic_checks

yaml = YAML(typ="safe")


class Activity(BaseModel):
    id: str = Field(..., alias="activity_id")
    name: str
    is_abstract: bool = False
    extends: list[str] | None = None
    defaults: dict | None = None
    allowed_actions: list[str] | None = None
    rules: dict | None = None

    # NEW: parsed Sequence DSL as an AST (list of Step nodes)
    sequence_ast: List[Dict[str, Any]] | None = None

    # NEW: results of semantic checks on the sequence AST
    sequence_checks: List[Dict[str, Any]] | None = None


# Prefer the canonical path, but support your older path too.
_PREFERRED_ROOTS = [
    Path("grammar/sports/squash"),        # new location
    Path("data/grammar/sports/squash"),   # legacy location
]


def _resolve_catalog_root() -> Path:
    for p in _PREFERRED_ROOTS:
        if p.exists():
            return p
    # Fallback to the first path (even if missing); caller can override.
    return _PREFERRED_ROOTS[0]


def _iter_yaml_files(root: Path) -> Iterable[Path]:
    # Only .yaml per your repo; add .yml if needed
    yield from root.rglob("*.yaml")


def _parse_rules_sequence_field(node: dict, file: Path, act_id: str) -> List[Dict[str, Any]] | None:
    rules = node.get("rules") or {}
    if not isinstance(rules, dict):
        return None
    seq = rules.get("sequence")
    if seq is None:
        return None
    try:
        return parse_rules_sequence(seq)  # accepts str OR list[str]
    except DSLParseError as e:
        # Enrich with file + activity id for easy debugging
        # Your DSLParseError likely carries (line, col, msg); we keep it robust.
        line = getattr(e, "line", None)
        col = getattr(e, "col", None)
        msg = getattr(e, "msg", str(e))
        where = f"{file} [{act_id}]"
        if line is not None and col is not None:
            raise DSLParseError(line, col, f"{where}: {msg}")
        raise DSLParseError(1, 1, f"{where}: {msg}")


def _effective_actions_policy(node: dict, default_policy: str) -> str:
    """
    Allow per-node override via:
      rules.allowed_actions_policy: "exact" | "family" | "exact_or_family"
    """
    rules = node.get("rules") or {}
    policy = rules.get("allowed_actions_policy")
    if policy in {"exact", "family", "exact_or_family"}:
        return policy
    return default_policy


def _nodes_from_exercise_variants(doc: dict, file: Path) -> Iterable[dict]:
    """
    Adapt the 'exercises/*.yaml' shape (family + variants) into activity-like nodes.
    - activity_id := "{family_id or family or file-stem}.{variant_id}"
    - name        := "{family}: {variant name}"
    - allowed_actions / rules come from the variant
    """
    family_name = doc.get("family") or file.stem
    family_id = doc.get("family_id") or f"squash.family.{family_name.lower().replace(' ', '_')}"
    variants = doc.get("variants") or []
    for v in variants:
        variant_id = v.get("variant_id") or "variant"
        activity_id = f"{family_id}.{variant_id}"
        name = v.get("name") or variant_id
        node = {
            "activity_id": activity_id,
            "name": f"{family_name}: {name}",
            "is_abstract": False,
            "extends": None,
            "defaults": None,
            "allowed_actions": v.get("allowed_actions"),
            "rules": v.get("rules"),
        }
        yield node


def _iter_activity_nodes(doc: dict, file: Path) -> Iterable[dict]:
    """
    Support both legacy ('activities'/'content') and new ('variants') file shapes.
    """
    # Legacy collections
    for key in ("activities", "content"):
        if key in doc and isinstance(doc[key], list):
            for node in doc[key]:
                if isinstance(node, dict):
                    yield node
            return

    # New exercise format with variants
    if "variants" in doc and isinstance(doc["variants"], list):
        yield from _nodes_from_exercise_variants(doc, file)
        return

    # Otherwise: nothing to load (templates/defaults/etc.)


def load_yaml_dir(
    root: Path | None = None,
    *,
    fail_on_hard: bool = True,
    default_actions_policy: str = "exact_or_family",
) -> dict[str, Activity]:
    """
    Load all activities under the catalog root, parse their rules.sequence (if present),
    run semantic checks, and return a dict keyed by activity_id.

    Args:
        root: override the detected catalog root.
        fail_on_hard: if True, raise on any hard check failure.
        default_actions_policy:
            - "exact"            -> names must match allowed_actions exactly
            - "family"           -> only the shot family must match (e.g., 'drive' matches 'hard drive')
            - "exact_or_family"  -> accept exact match OR same family (recommended)
    """
    catalog_root = root or _resolve_catalog_root()
    acts: dict[str, Activity] = {}

    for file in _iter_yaml_files(catalog_root):
        doc = yaml.load(file.read_text()) or {}

        for node in _iter_activity_nodes(doc, file):
            if not isinstance(node, dict):
                continue
            if "activity_id" not in node:  # e.g., template node
                continue

            # Build the Activity (pydantic handles aliases)
            act = Activity.model_validate(node)

            # Parse the DSL if present
            seq_ast = _parse_rules_sequence_field(node, file, act.id)
            act.sequence_ast = seq_ast

            # Run semantic checks if we have a sequence
            if seq_ast:
                policy = _effective_actions_policy(node, default_actions_policy)
                allowed = act.allowed_actions or []
                results = run_semantic_checks(seq_ast, allowed_actions=allowed, policy=policy)
                # Store as plain dicts (easy to serialize / log)
                act.sequence_checks = [
                    {
                        "check": r.check,
                        "ok": r.ok,
                        "severity": r.severity,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in results
                ]

                if fail_on_hard and any((not r["ok"]) and r["severity"] == "hard" for r in act.sequence_checks):
                    msgs = "; ".join(
                        f"{r['check']}: {r['message']}" for r in act.sequence_checks if (not r["ok"]) and r["severity"] == "hard"
                    )
                    raise ValueError(f"{file} [{act.id}] hard semantic errors -> {msgs}")

            acts[act.id] = act

    return acts

# --- CLI ---------------------------------------------------------------
def _to_plain(activity: Activity) -> dict:
    return {
        "activity_id": activity.id,
        "name": activity.name,
        "has_sequence": activity.sequence_ast is not None,
        "checks": activity.sequence_checks or [],
    }


def _print_text(errors: list[dict], *, show_soft: bool, show_hard: bool) -> None:
    if not errors:
        print("No issues found.")
        return
    for row in errors:
        aid = row["activity_id"]
        name = row["name"]
        for chk in row["checks"]:
            if not chk.get("ok", True):
                sev = chk.get("severity", "soft")
                if (sev == "soft" and not show_soft) or (sev == "hard" and not show_hard):
                    continue
                msg = chk.get("message", "")
                ctype = chk.get("check", "")
                print(f"[{sev.upper()}] {aid} :: {name} :: {ctype} -> {msg}")


def _print_summary(acts: dict[str, Activity]) -> None:
    total_with_seq = sum(1 for a in acts.values() if a.sequence_ast)
    total_acts = len(acts)
    hard_fail = 0
    soft_fail = 0
    for a in acts.values():
        for chk in a.sequence_checks or []:
            if not chk.get("ok", True):
                if chk.get("severity") == "hard":
                    hard_fail += 1
                else:
                    soft_fail += 1
    print("=== Summary ===")
    print(f"activities: {total_acts}")
    print(f"with sequence: {total_with_seq}")
    print(f"failed checks (hard): {hard_fail}")
    print(f"failed checks (soft): {soft_fail}")


def main():
    import argparse, json, sys

    parser = argparse.ArgumentParser(
        description="Validate DSL sequences and semantic constraints across the grammar catalog."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override catalog root (defaults to grammar/sports/squash or data/grammar/sports/squash).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="exact_or_family",
        choices=["exact", "family", "exact_or_family"],
        help="Allowed-actions matching policy.",
    )
    parser.add_argument(
        "--list-errors",
        action="store_true",
        help="List only failing checks (soft + hard).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a pass/fail summary.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json", "jsonl"],
        help="Output format for --list-errors (text/json/jsonl).",
    )
    parser.add_argument(
        "--include-soft",
        action="store_true",
        help="When listing errors, include soft failures (default includes both in text mode; for json/jsonl you can filter downstream).",
    )
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help="When listing errors, include hard failures (on by default).",
    )
    parser.add_argument(
        "--no-hard-exit",
        action="store_true",
        help="Do not use non-zero exit code when hard failures exist.",
    )
    args = parser.parse_args()

    # Sensible defaults for which severities to show
    show_soft = args.include_soft or (args.format == "text")
    show_hard = True if not args.include_hard else args.include_hard

    root = Path(args.root) if args.root else None

    # We never raise on hard here; we want to report them instead.
    acts = load_yaml_dir(
        root=root,
        fail_on_hard=False,
        default_actions_policy=args.policy,
    )

    # Build a lightweight view
    rows = [_to_plain(a) for a in acts.values()]

    # Filter to only failing checks for printing
    def has_fail(d: dict, sev: Optional[str] = None) -> bool:
        for c in d.get("checks", []):
            if not c.get("ok", True):
                if sev is None or c.get("severity") == sev:
                    return True
        return False

    hard_fail_present = any(has_fail(r, "hard") for r in rows)

    if args.list_errors:
        failing = []
        for r in rows:
            fails = [c for c in r["checks"] if not c.get("ok", True)]
            if not fails:
                continue
            # Apply severity filter for text mode
            if args.format == "text":
                r = {
                    "activity_id": r["activity_id"],
                    "name": r["name"],
                    "checks": [
                        c for c in fails
                        if (c.get("severity") != "soft" or show_soft)
                        and (c.get("severity") != "hard" or show_hard)
                    ],
                }
                if not r["checks"]:
                    continue
                _print_text([r], show_soft=show_soft, show_hard=show_hard)
            elif args.format == "json":
                failing.append({
                    "activity_id": r["activity_id"],
                    "name": r["name"],
                    "checks": fails,
                })
            else:  # jsonl
                for c in fails:
                    print(json.dumps({
                        "activity_id": r["activity_id"],
                        "name": r["name"],
                        "check": c,
                    }))
        if args.format == "json" and failing:
            import json as _json
            print(_json.dumps(failing, indent=2))

    if args.summary or (not args.list_errors):
        _print_summary(acts)

    # Exit code for CI: 2 if hard failures unless --no-hard-exit
    if hard_fail_present and not args.no_hard_exit:
        sys.exit(2)


if __name__ == "__main__":
    main()
