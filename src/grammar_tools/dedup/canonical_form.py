# src/grammar_tools/dedup/canonical_form.py
"""
Canonicalise a session plan without losing information.
- Keep ALL blocks.
- Infer a stable block 'type' when missing.
- Order blocks deterministically by (type_order, explicit order, name, original_index).
- Do NOT reorder exercises inside blocks (sequence is semantically meaningful).
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple

# You can extend this if your grammar adds more block types.
TYPE_ORDER = {
    "warmup": 0,
    "warm-up": 0,
    "warm_up": 0,
    "activity": 1,
    "drill_block": 1,
    "game_block": 1,
    "cooldown": 2,
    "cool-down": 2,
    "cool_down": 2,
}

def _infer_block_type(block: Dict[str, Any]) -> str:
    # Prefer explicit metadata
    t = (block.get("type") or block.get("kind") or "").strip().lower()
    if t:
        return t
    # Fall back to name heuristics (robust to hyphen/case variants)
    name = (block.get("name") or "").strip().lower()
    if "warm" in name and "up" in name:
        return "warmup"
    if "cool" in name and "down" in name:
        return "cooldown"
    return "activity"

def to_canonical_plan(session_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a canonical plan:
      - deep copy
      - normalize block names (strip)
      - add a private _ctype used for stable sorting (removed before return)
    """
    plan = copy.deepcopy(session_plan)
    blocks: List[Dict[str, Any]] = plan.get("blocks", [])
    indexed: List[Tuple[int, Dict[str, Any]]] = list(enumerate(blocks))

    for _, b in indexed:
        # Normalize display fields with zero semantics
        if isinstance(b.get("name"), str):
            b["name"] = b["name"].strip()
        # Compute canonical type
        b["_ctype"] = _infer_block_type(b)

    def sort_key(idx_and_block: Tuple[int, Dict[str, Any]]):
        i, b = idx_and_block
        # use explicit numeric order if present, else large sentinel
        explicit_order = b.get("order")
        if not isinstance(explicit_order, (int, float)):
            explicit_order = 10**9
        return (
            TYPE_ORDER.get(b["_ctype"], 50),
            explicit_order,
            b.get("name", ""),
            i,  # original index as final tiebreaker
        )

    indexed.sort(key=sort_key)
    plan["blocks"] = [b for _, b in indexed]

    # Remove helper keys
    for b in plan["blocks"]:
        if "_ctype" in b:
            del b["_ctype"]

    return plan
