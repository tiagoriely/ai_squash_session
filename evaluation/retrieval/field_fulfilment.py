"""
Metric helpers for retrieval experiments.

Now considers **primaryShots** and **secondaryShots** when checking the
`shots` requirement, so the metrics reflect what is really inside the
documents.

Functions
---------
best_hit(ranked_docs, user)
mean_k(ranked_docs, user, k=5)
coverage_k(ranked_docs, user, k=5)
"""

from __future__ import annotations
from typing import Mapping, Sequence, Any

from rag.pipelines.retrieval.field_retrieval.field_matcher import (
    clean_and_standardise_value,
)


# ------------------------------------------------------------------ #

def _doc_shot_set(doc: Mapping[str, Any]) -> set[str]:
    """Return **canonical** union of shots, primaryShots and secondaryShots."""
    raw: list[str] = []
    for f in ("shots", "primaryShots", "secondaryShots"):
        raw.extend(doc.get(f, []))
    return {clean_and_standardise_value("shots", s) for s in raw}


def _field_is_satisfied(field: str, u_val: Any, doc: Mapping[str, Any]) -> bool:
    """Single‑field check mirroring the logic used in the original demo."""

    # ----- shots (union of all shot lists) ------------------------ #
    if field == "shots":
        u_set = {
            clean_and_standardise_value("shots", s)
            for s in (u_val if isinstance(u_val, list) else [u_val])
        }
        return u_set.issubset(_doc_shot_set(doc))

    d_val = doc.get(field)

    # ----- list‑valued fields ------------------------------------- #
    if isinstance(u_val, list):
        u_set = {clean_and_standardise_value(field, x) for x in u_val}
        d_set = {clean_and_standardise_value(field, x) for x in (d_val or [])}
        return u_set.issubset(d_set)

    # ----- numeric tolerance for duration ------------------------- #
    if field == "duration":
        try:
            return abs(int(u_val) - int(d_val)) <= 10
        except (TypeError, ValueError):
            return False

    # ----- scalar fields ------------------------------------------ #
    u_std = clean_and_standardise_value(field, u_val)
    d_std = clean_and_standardise_value(field, d_val)
    return u_std == d_std


# ------------------------------------------------------------------ #

def fulfilment_score(
    doc: Mapping[str, Any],
    user: Mapping[str, Any],
    check_fields: tuple[str, ...] = (
        "type",
        "participants",
        "squashLevel",
        "intensity",
        "duration",
        "shots",
        "shotSide",
        "movement",
    ),
) -> float:
    """Fraction ∈ [0, 1] of *requested* fields satisfied by **doc**."""
    total = matched = 0
    for f in check_fields:
        u_val = user.get(f)
        if u_val is None:
            continue
        total += 1
        if _field_is_satisfied(f, u_val, doc):
            matched += 1
    return matched / total if total else 1.0


# ----------------------- aggregate helpers ------------------------ #

def best_hit(ranked: Sequence[Mapping[str, Any]], user) -> float:
    """Fulfilment of the top‑ranked document (rank‑1)."""
    return fulfilment_score(ranked[0], user) if ranked else 0.0



def mean_k(ranked: Sequence[Mapping[str, Any]], user, k: int = 5) -> float:
    """Average fulfilment across the first *k* docs."""
    if not ranked:
        return 0.0
    k = min(k, len(ranked))
    return sum(fulfilment_score(d, user) for d in ranked[:k]) / k



def coverage_k(ranked: Sequence[Mapping[str, Any]], user, k: int = 5) -> float:
    """Share of requested fields satisfied by **at least one** of the top‑k docs."""
    if not ranked:
        return 0.0

    covered = set()
    for field, u_val in user.items():
        for d in ranked[:k]:
            if _field_is_satisfied(field, u_val, d):
                covered.add(field)
                break
    return len(covered) / len(user)
