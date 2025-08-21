# rag/pipelines/generation/grammar_constraints_integration.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import yaml

# Use your DSL parser
from src.grammar_tools.dsl_tools.parser import parse_rules_sequence


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
# Root of the squash grammar (keep consistent with src/grammar_tools/planner.py)
_SQUASH_ROOT = Path("grammar") / "sports" / "squash"
_SESSION_TYPES_PATH = _SQUASH_ROOT / "sessions_types.yaml"  # file on disk
# Exercise families live here:
GRAMMAR_ROOT = _SQUASH_ROOT / "exercises"


# --------------------------------------------------------------------------- #
# Modes & dataclasses                                                         #
# --------------------------------------------------------------------------- #
class ConstraintsMode(str, Enum):
    SOFT = "soft"      # put rules into the prompt only
    HARD = "hard"      # post-hoc enforce/report only
    HYBRID = "hybrid"  # both: prompt + post-hoc report


@dataclass
class VariantConstraints:
    variant_id: str
    name: str
    allowed_actions: List[str]
    sequence_lines: List[str]          # raw authoring lines (for readable prompt)
    sequence_ast: List[Dict[str, Any]] # parsed DSL (for checks if needed)


@dataclass
class FamilyConstraints:
    family_id: str
    family_name: str
    variants: List[VariantConstraints]

    def allowed_actions_union(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in self.variants:
            for a in (v.allowed_actions or []):
                n = " ".join(a.lower().split())
                if n not in seen:
                    out.append(a)
                    seen.add(n)
        return out


# --------------------------------------------------------------------------- #
# YAML helpers                                                                #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    # Back-compat alias (some code above uses _read_yaml)
    return _load_yaml(path)


def _iter_family_files() -> List[Path]:
    # Skip warmups/defaults by name
    return [
        p for p in sorted(GRAMMAR_ROOT.glob("*.yaml"))
        if p.name.lower() not in {"warmups.yaml", "defaults.yaml"}
    ]


def _id_stem(any_id: str) -> str:
    """
    Accepts either a bare file-stem (e.g. 'conditioned_game_v1') or a dotted id
    (e.g. 'squash.structure.conditioned_game_v1') and returns the trailing token.
    """
    if not any_id:
        raise ValueError("empty id")
    return any_id.split(".")[-1]


# --------------------------------------------------------------------------- #
# Family constraints (for prompting & enforcement)                            #
# --------------------------------------------------------------------------- #
def _load_family_by_id(target_family_id: str | None) -> Tuple[Dict[str, Any], Path]:
    """
    If target_family_id is given, find the YAML whose 'family_id' matches.
    Else, pick the first family file to keep things simple.
    """
    files = _iter_family_files()
    if not files:
        raise FileNotFoundError(f"No exercise family YAML found under {GRAMMAR_ROOT}")

    if not target_family_id:
        p = files[0]
        return _read_yaml(p), p

    for p in files:
        doc = _read_yaml(p)
        if doc.get("family_id") == target_family_id:
            return doc, p

    # fallback: try to match by filename end
    for p in files:
        if p.stem.replace("_", ".") in (target_family_id or ""):
            return _read_yaml(p), p

    raise FileNotFoundError(
        f"Could not find a family YAML with family_id='{target_family_id}'."
    )


def _as_list_of_lines(seq_val: Any) -> List[str]:
    if seq_val is None:
        return []
    if isinstance(seq_val, str):
        return [seq_val]
    if isinstance(seq_val, list):
        # ensure str
        return [str(x) for x in seq_val]
    return [str(seq_val)]


def load_family_constraints(target_family_id: str | None = None) -> FamilyConstraints:
    """
    Load a squash exercise family YAML and return constraints useful for prompting/enforcement.
    """
    doc, path = _load_family_by_id(target_family_id)
    fam_name = doc.get("family", path.stem.replace("_", " ").title())
    fam_id = doc.get("family_id", path.stem)

    variants: List[VariantConstraints] = []
    for v in doc.get("variants", []):
        rules = v.get("rules", {}) or {}
        seq_lines = _as_list_of_lines(rules.get("sequence"))
        # parse each line and flatten
        seq_ast_all: List[Dict[str, Any]] = []
        for line in seq_lines:
            seq_ast_all.extend(parse_rules_sequence(line))

        allowed = v.get("allowed_actions") or rules.get("allowed_actions") or []

        variants.append(
            VariantConstraints(
                variant_id=v.get("variant_id", "v0"),
                name=v.get("name", "Variant"),
                allowed_actions=list(allowed),
                sequence_lines=seq_lines,
                sequence_ast=seq_ast_all,
            )
        )

    return FamilyConstraints(family_id=fam_id, family_name=fam_name, variants=variants)


def build_constraints_block(
    fam: FamilyConstraints,
    mode: ConstraintsMode = ConstraintsMode.HYBRID,
    include_sequences: bool = True,
) -> str:
    """
    Produce a compact, LLM-friendly block that:
      • lists the allowed actions (union)
      • (optionally) shows the canonical sequences per variant
    """
    lines: List[str] = []
    lines.append("### Constraints (follow strictly)")
    lines.append("")
    lines.append(
        "Only use these shots/labels when writing the plan (use synonyms only if they map to these):"
    )
    allowed = fam.allowed_actions_union()
    if allowed:
        lines.append("- Allowed actions: " + ", ".join(sorted(set(allowed))))
    else:
        lines.append("- Allowed actions: (not specified)")

    if include_sequences and fam.variants:
        lines.append("")
        lines.append("Respect these canonical sequences (per variant):")
        for v in fam.variants:
            if not v.sequence_lines:
                continue
            # join multiple lines into a single compact bullet to save tokens
            seq_compact = "  →  ".join(l for l in v.sequence_lines if l.strip())
            lines.append(f"- {v.name}: {seq_compact}")

    if mode == ConstraintsMode.SOFT:
        lines.append("")
        lines.append("Interpretation mode: SOFT (instructions guide the plan; no post-hoc edits).")
    elif mode == ConstraintsMode.HARD:
        lines.append("")
        lines.append("Interpretation mode: HARD (outputs will be post-checked; keep strictly compliant).")
    else:
        lines.append("")
        lines.append("Interpretation mode: HYBRID (instructions + post-check).")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Session STRUCTURE helpers (prompt + light validation)                        #
# --------------------------------------------------------------------------- #
def load_structure(structure_id: str) -> Dict[str, Any]:
    """
    Load a session-structure YAML by id.

    Examples:
      load_structure("conditioned_game_v1")
      load_structure("squash.structure.conditioned_game_v1")
    """
    stem = _id_stem(structure_id)
    path = _SQUASH_ROOT / "session_structures" / f"{stem}.yaml"
    struct = _load_yaml(path)
    struct.setdefault("_file", str(path))
    struct.setdefault("_id", structure_id)
    return struct


def _resolve_minutes(d) -> float:
    """
    Accepts: number | {"target": number} | anything else → returns a number (fallback 0).
    Mirrors the planner's permissiveness.
    """
    if isinstance(d, dict):
        return float(d.get("target", 0) or 0)
    try:
        return float(d)
    except Exception:
        return 0.0


def build_structure_block(struct: Dict[str, Any]) -> str:
    """
    Turn a structure YAML into a compact prompt appendix the LLM can follow.
    Only includes info that is fairly universal in your files: blocks + slot durations.
    If a block has `block_type`, we surface it as a hint for the model.
    """
    lines: List[str] = []
    lines.append("### STRUCTURE SPEC")
    tgt = struct.get("target_duration_minutes")
    if tgt:
        lines.append(f"- Target duration: ~{tgt} min")
    lines.append("- Build the session using these blocks IN ORDER:")

    for b in struct.get("blocks", []):
        bname = b.get("block_name", "Block")
        slots = b.get("exercises", [])
        btype = b.get("block_type")
        # exercise slot durations (resolve dict target)
        slot_mins = [_resolve_minutes(s.get("duration_minutes")) for s in slots]
        if slot_mins:
            mins_str = ", ".join(
                f"{m:g} min" if float(m).is_integer() else f"{m} min" for m in slot_mins
            )
            lines.append(f"  • {bname}: slot durations = [{mins_str}]")
        else:
            lines.append(f"  • {bname}: (no per-slot durations specified)")
        if btype:
            lines.append(f"    (session type for this block: {btype})")

    lines.append("- Keep the headings for each block in the output.")
    return "\n".join(lines)


def validate_structure(generated: str, struct: Dict[str, Any]) -> List[str]:
    text = generated or ""
    errs: List[str] = []

    # 1) Warm-up must appear
    want_warmup = any((b.get("block_name") or "").lower().startswith("warm-up") for b in struct.get("blocks", []))
    if want_warmup and re.search(r"\bwarm-?up\b", text, re.I) is None:
        errs.append("Missing block heading: 'Warm-up'")

    # 2) Count of Activity-like blocks; accept either “Activity Block X” or “Conditioned Game X”
    activity_blocks = [b for b in struct.get("blocks", []) if "Activity" in (b.get("block_name") or "")]
    if activity_blocks:
        cg_found = len(re.findall(r"\bConditioned Game\b", text, re.I))
        ab_found = len(re.findall(r"\bActivity Block\b", text, re.I))
        if (cg_found + ab_found) != len(activity_blocks):
            errs.append(f"Expected {len(activity_blocks)} activity sections, found {cg_found + ab_found} "
                        "(counting 'Activity Block' or 'Conditioned Game').")

    return errs


# --------------------------------------------------------------------------- #
# Session ARCHETYPE helpers (prompt + light validation)                        #
# --------------------------------------------------------------------------- #
def load_archetype(archetype_id: str) -> Dict[str, Any]:
    """
    Load an archetype YAML by id.

    Examples:
      load_archetype("progressive_family")
      load_archetype("squash.archetype.progressive_family")
    """
    stem = _id_stem(archetype_id)
    path = _SQUASH_ROOT / "session_archetypes" / f"{stem}.yaml"
    arch = _load_yaml(path)
    arch.setdefault("_file", str(path))
    arch.setdefault("_id", archetype_id)
    return arch


def build_archetype_block(arch: Dict[str, Any]) -> str:
    """
    Turn an archetype YAML into a prompt appendix of 'progression/flow' rules.
    Falls back to generic guidance if the file is sparse.
    """
    lines: List[str] = []
    lines.append("### ARCHETYPE RULES")
    name = arch.get("archetype_name") or arch.get("name") or _id_stem(arch.get("_id", "archetype"))
    lines.append(f"- Archetype: {name}")

    # Common fields you typically have; optional & order-agnostic.
    if arch.get("description"):
        lines.append(f"- Description: {arch['description']}")

    if arch.get("progression_order"):
        po = arch["progression_order"]
        if isinstance(po, list) and po:
            lines.append(f"- Prefer variant order: {', '.join(map(str, po))}")

    if arch.get("rules") and isinstance(arch["rules"], (list, tuple)):
        lines.append("- Rules:")
        for r in arch["rules"]:
            lines.append(f"  • {r}")

    # Generic fallback
    if len(lines) <= 2:
        lines.append("- Use increasing difficulty/complexity across Activity blocks.")
        lines.append("- Keep roles/sides consistent within a block; alternate sides across blocks if relevant.")

    return "\n".join(lines)


def validate_archetype(generated: str, arch: Dict[str, Any]) -> List[str]:
    """
    Lightweight checks, designed to be non-brittle:
      - If archetype hints at side alternation, require both 'forehand' and 'backhand' in text.
      - If a fixed progression_order is given, ensure at least their names/ids appear in order.
    """
    text = generated or ""
    errs: List[str] = []

    # Check for sides if requested
    wants_sides = False
    for k in ("rules", "description"):
        v = arch.get(k)
        if isinstance(v, str) and re.search(r"\bforehand\b.*\bbackhand\b|\bbackhand\b.*\bforehand\b", v, re.I):
            wants_sides = True
        if isinstance(v, list) and any(re.search(r"\bforehand\b|\bbackhand\b", str(x), re.I) for x in v):
            wants_sides = True

    if wants_sides:
        has_fh = re.search(r"\bforehand\b", text, re.I)
        has_bh = re.search(r"\bbackhand\b", text, re.I)
        if not (has_fh and has_bh):
            errs.append("Archetype expects both 'forehand' and 'backhand' to appear in the plan.")

    # If a strict progression_order is given, check they appear in sequence (loose substring check).
    po = arch.get("progression_order")
    if isinstance(po, list) and len(po) >= 2:
        idx = 0
        for token in map(str, po):
            m = re.search(re.escape(token), text, re.I)
            if not m:
                errs.append(f"Variant from progression_order not mentioned: '{token}'")
                # keep going to list all missing ones
            else:
                # ensure monotonic increasing positions (very loose ordering check)
                if m.start() < idx:
                    errs.append("Progression order appears out of sequence in the text.")
                    break
                idx = m.start()

    return errs


# --------------------------------------------------------------------------- #
# Session TYPE helpers (prompt + light validation)                             #
# --------------------------------------------------------------------------- #
def load_session_type(session_type_id: str) -> Dict[str, Any]:
    """
    Load a session-type spec by id (e.g., 'conditioned_game', 'drill', 'solo', 'ghosting').
    Expected file: grammar/sports/squash/sessions_types.yaml with structure:
      types:
        conditioned_game:
          name: "Conditioned Game"
          heading_label: "Conditioned Game"
          defaults: { points_max: 11, roles_switch: true, sides_switch: true }
          rules:
            - "roles must switch halfway"
          output_hints:
            - "Each item should show points like 'First to 7/9/11 points'"
    """
    try:
        data = _load_yaml(_SESSION_TYPES_PATH)
    except FileNotFoundError:
        data = {}
    tmap = (data.get("types") or {}) if isinstance(data, dict) else {}

    st = (tmap.get(session_type_id) or {}) if isinstance(tmap, dict) else {}
    if not st:
        # fallback minimal spec
        return {
            "_id": session_type_id,
            "_file": str(_SESSION_TYPES_PATH),
            "name": session_type_id.title(),
            "heading_label": session_type_id.title(),
            "defaults": {},
            "rules": [],
            "output_hints": [],
        }

    st = dict(st)  # shallow copy
    st.setdefault("_id", session_type_id)
    st.setdefault("_file", str(_SESSION_TYPES_PATH))
    st.setdefault("name", session_type_id.title())
    st.setdefault("heading_label", st.get("name", session_type_id.title()))
    st.setdefault("defaults", {})
    st.setdefault("rules", [])
    st.setdefault("output_hints", [])
    return st


def build_session_type_block(st: Dict[str, Any]) -> str:
    """
    Render a compact 'TYPE RULES' appendix the model can follow.
    Keeps it short to avoid overpowering Context.
    """
    lines: List[str] = []
    lines.append("### TYPE RULES")
    lines.append(f"- Session type: {st.get('name') or st.get('heading_label')}")
    hl = st.get("heading_label")
    if hl:
        lines.append(f"- Use the heading label: '{hl}' for each section of this type.")
    # Defaults
    d = st.get("defaults") or {}
    if "points_max" in d:
        try:
            lines.append(f"- Points cap per item: ≤ {int(d['points_max'])} (e.g., 'First to 7/9/11 points').")
        except Exception:
            pass
    if (d.get("roles_switch") is True):
        lines.append("- Roles must switch halfway unless explicitly stated otherwise.")
    if (d.get("sides_switch") is True):
        lines.append("- Court side (forehand/backhand) must switch after the role switch.")
    # Rules/hints
    for r in (st.get("rules") or []):
        lines.append(f"- {r}")
    if st.get("output_hints"):
        lines.append("- Output hints:")
        for h in st["output_hints"]:
            lines.append(f"  • {h}")
    return "\n".join(lines)


def validate_session_type(generated: str, st: Dict[str, Any]) -> List[str]:
    """
    Gentle, text-based checks:
      - heading label appears at least once
      - points cap not exceeded
      - role/sides switching hints (regex heuristic)
    """
    errs: List[str] = []
    text = generated or ""

    # Heading presence
    hl = (st.get("heading_label") or "").strip()
    if hl:
        if re.search(re.escape(hl), text, re.I) is None:
            errs.append(f"Missing session-type heading label: '{hl}'")

    # points cap
    d = st.get("defaults") or {}
    try:
        pmax = int(d.get("points_max", 0) or 0)
    except Exception:
        pmax = 0
    if pmax:
        pts = [int(m.group(1))
               for m in re.finditer(r"\b(?:first\s+to\s+)?(\d{1,2})\s+points?\b", text, re.I)]
        if any(x > pmax for x in pts):
            errs.append(f"Found an item exceeding points cap ({pmax}).")

    # role switch heuristic
    if d.get("roles_switch") is True:
        if (re.search(r"\bswitch(ing)?\s+roles?\b", text, re.I) is None and
            re.search(r"\bswap\s+roles?\b", text, re.I) is None):
            errs.append("Type expects a role switch; not detected in the text.")

    # side switch heuristic
    if d.get("sides_switch") is True:
        has_fh = re.search(r"\bforehand\b", text, re.I)
        has_bh = re.search(r"\bbackhand\b", text, re.I)
        if not (has_fh and has_bh):
            errs.append("Type expects forehand/backhand coverage; not both detected.")

    return errs
