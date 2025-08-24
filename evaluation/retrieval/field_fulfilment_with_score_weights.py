"""
Weighted Field‑Fulfilment helpers
================================

These utilities mirror *field_fulfilment.py* but multiply every field by
its importance weight taken from ``FIELD_SCORING_CONFIG``.  The idea is
simple:

    weighted_score = Σ  w_f * 1[field satisfied]
                     ───────────────────────────────
                     Σ  w_f (only for *requested* fields)

Thus a miss on a high‑weight field (e.g. ``shots``) hurts much more than
on a low‑weight field (e.g. ``duration``).

Functions
---------
* ``weighted_score(doc, user)`` ― fulfilment ∈ [0,1]
* ``best_hit_w(ranked, user)`` ― score of top‑ranked doc
* ``mean_k_w(ranked, user, k=5)`` ― average across first *k*
* ``coverage_k_w(ranked, user, k=5)`` ― share of *weighted* request that
  is satisfied by at least one of the top‑k documents

All helpers keep **exactly the same tolerance rules** as the un‑weighted
version so the two can be compared directly.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Any

from rag.pipelines.retrieval.field_retrieval.field_matcher import (
    clean_and_standardise_value,
)
from archive.rag.retrieval.field_retrieval_config import FIELD_SCORING_CONFIG

# ---------------------------------------------------------------------
# low‑level satisfaction check – identical to the plain version
# ---------------------------------------------------------------------

def _field_is_satisfied(field: str, u_val, d_val) -> bool:
    """Return *True* iff the document value satisfies the user value."""
    # ---- list fields -------------------------------------------------
    if isinstance(u_val, list):
        u_set = {clean_and_standardise_value(field, x) for x in u_val}
        d_set = set(clean_and_standardise_value(field, d_val or []))
        return u_set.issubset(d_set)

    # ---- numeric tolerance for duration -----------------------------
    if field == "duration":
        try:
            return abs(int(u_val) - int(d_val)) <= 10
        except (TypeError, ValueError):
            return False

    # ---- scalar fields ----------------------------------------------
    u_std = clean_and_standardise_value(field, u_val)
    d_std = clean_and_standardise_value(field, d_val)
    return u_std == d_std


# ---------------------------------------------------------------------
# top‑level scorers
# ---------------------------------------------------------------------

def _weight(field: str) -> float:
    """Return the *base_weight* from FIELD_SCORING_CONFIG or 1.0."""
    cfg = FIELD_SCORING_CONFIG.get(field, {})
    return float(cfg.get("base_weight", 1.0))


def weighted_score(
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
        "primaryShots",
        "secondaryShots",
    ),
) -> float:
    """Weighted fulfilment ∈ [0,1] for *this* document."""
    total_w = matched_w = 0.0

    for f in check_fields:
        u_val = user.get(f)
        if u_val is None:
            continue  # field not requested

        w = _weight(f)
        total_w += w
        if _field_is_satisfied(f, u_val, doc.get(f)):
            matched_w += w

    return matched_w / total_w if total_w else 1.0


# ------------------------------------------------------------------
# aggregate helpers – analogue to best_hit / mean_k / coverage_k
# ------------------------------------------------------------------

def best_hit_w(ranked: Sequence[Mapping[str, Any]], user) -> float:
    """Weighted fulfilment of the first result."""
    return weighted_score(ranked[0], user) if ranked else 0.0


def mean_k_w(ranked: Sequence[Mapping[str, Any]], user, k: int = 5) -> float:
    """Average weighted fulfilment across top *k* docs."""
    if not ranked:
        return 0.0
    k = min(k, len(ranked))
    return sum(weighted_score(d, user) for d in ranked[:k]) / k


def coverage_k_w(ranked: Sequence[Mapping[str, Any]], user, k: int = 5) -> float:
    """Weighted coverage: Σ_w(fields covered by ≥1 of top‑k) / Σ_w(requested)."""
    if not ranked:
        return 0.0

    requested_fields = {f for f in user if user[f] is not None}
    covered_w = 0.0
    total_w = sum(_weight(f) for f in requested_fields)

    for f in requested_fields:
        for d in ranked[:k]:
            if _field_is_satisfied(f, user[f], d.get(f)):
                covered_w += _weight(f)
                break

    return covered_w / total_w if total_w else 1.0
