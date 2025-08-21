# src/grammar_tools/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ruamel.yaml import YAML
from pydantic import BaseModel, Field

# DSL parser and semantic checks
from src.grammar_tools.dsl_tools.parser import parse_rules_sequence, ParseError as DSLParseError
from src.grammar_tools.dsl_tools.semantic_checks import run_semantic_checks

yaml = YAML(typ="safe")


class Activity(BaseModel):
    id: str = Field(..., alias="activity_id")
    name: str
    family_id: Optional[str] = None
    variant_id: Optional[str] = None
    is_abstract: bool = False
    extends: list[str] | None = None
    defaults: dict | None = None
    allowed_actions: list[str] | None = None
    rules: dict | None = None
    sequence_ast: List[Dict[str, Any]] | None = None
    sequence_checks: List[Dict[str, Any]] | None = None
    types: List[str] = []
    category: Optional[str] = None
    family: Optional[str] = None
    parameters: List[Dict[str, Any]] = []
    shots: Optional[Dict[str, List[str]]] = None
    movement: Optional[List[str]] = None
    foundational_shots: List[str] = []
    tactical_shots: List[str] = []
    difficulty_score: Optional[int] = None


_PREFERRED_ROOTS = [Path("grammar/sports/squash"), Path("data/grammar/sports/squash")]


def _resolve_catalog_root() -> Path:
    for p in _PREFERRED_ROOTS:
        if p.exists():
            return p
    return _PREFERRED_ROOTS[0]


def _iter_yaml_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.yaml")


def _parse_rules_sequence_field(node: dict, file: Path, act_id: str) -> List[Dict[str, Any]] | None:
    rules = node.get("rules") or {}
    if not isinstance(rules, dict):
        return None
    seq = rules.get("sequence")
    if seq is None:
        return None
    try:
        return parse_rules_sequence(seq)
    except DSLParseError as e:
        msg = getattr(e, "msg", str(e))
        raise DSLParseError(1, 1, f"{file} [{act_id}]: {msg}") from e


def _effective_actions_policy(node: dict, default_policy: str) -> str:
    rules = node.get("rules") or {}
    policy = rules.get("allowed_actions_policy")
    if policy in {"exact", "family", "exact_or_family"}:
        return policy
    return default_policy


def _iter_activity_nodes(doc: Any, file: Path) -> Iterable[dict]:
    if not isinstance(doc, dict) or "variants" not in doc:
        return

    family_id = doc.get("family_id", f"squash.family.{file.stem}")
    for variant in doc.get("variants", []):
        # 1. First, make a copy of the entire variant dictionary to preserve all keys.
        node = variant.copy()

        # 2. Then, update it with the family-level and generated information.
        vid = variant.get("variant_id", "default")
        node.update({
            "activity_id": f"{family_id}.{vid}",
            "family_id": family_id,
            "variant_id": vid,
            "name": f"{doc.get('family', file.stem)}: {variant.get('name', vid)}",
            "category": doc.get("category"),
            "family": doc.get("family"),
            "parameters": doc.get("parameters", [])
        })
        yield node


def load_yaml_dir(
        root: Path | None = None,
        *,
        fail_on_hard: bool = True,
        default_actions_policy: str = "exact_or_family",
) -> dict[str, Activity]:
    catalog_root = root or _resolve_catalog_root()
    acts: dict[str, Activity] = {}

    for file in _iter_yaml_files(catalog_root):
        try:
            doc = yaml.load(file.read_text())
            if doc is None:
                continue

            for node in _iter_activity_nodes(doc, file):
                if "activity_id" not in node:
                    continue

                # Use model_validate to handle extra fields gracefully
                act = Activity.model_validate(node)
                seq_ast = _parse_rules_sequence_field(node, file, act.id)
                act.sequence_ast = seq_ast

                if seq_ast:
                    policy = _effective_actions_policy(node, default_actions_policy)
                    allowed = act.allowed_actions or []
                    results = run_semantic_checks(seq_ast, allowed_actions=allowed, policy=policy)

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

                    if fail_on_hard and any(not r['ok'] and r['severity'] == "hard" for r in act.sequence_checks or []):
                        raise ValueError(f"{file} [{act.id}] has hard semantic errors.")

                acts[act.id] = act
        except Exception as e:
            print(f"⚠️  Could not process file {file}: {e}")

    return acts