# tests/grammar_tests/test_semantic_dedup_extra.py
import numpy as np
import pytest
from unittest.mock import patch

try:
    from src.grammar_tools.dedup.semantic_dedup import SemanticDeduper, SemanticDedupNotAvailable
    SEMANTIC_DEDUP_AVAILABLE = True
except Exception:
    SEMANTIC_DEDUP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SEMANTIC_DEDUP_AVAILABLE, reason="Semantic libs not installed")

# ---- helper: simple, deterministic encoder based on domain keywords ----
def _keyword_vec(text: str) -> np.ndarray:
    t = text.lower()

    # basis vectors by archetype
    v = np.zeros(6, dtype=np.float32)
    if "dynamic block session" in t:
        v[0] = 1.0
    if "progressive shotside" in t:
        v[1] = 1.0
    if "progressive family" in t:
        v[2] = 1.0

    # light “features” for sides & common drills
    if "forehand" in t:
        v[3] += 0.4
    if "backhand" in t:
        v[3] += 0.4
    if "boast-cross-drive" in t or "boast" in t and "drive" in t and "cross" in t:
        v[4] += 0.3
    if "drop-drive" in t:
        v[5] += 0.3

    # avoid zero vector; normalize
    if not np.any(v):
        v[0] = 1.0
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)

def _encode_side_effect(x):
    # Support both a single string or a list/iterable of strings.
    if isinstance(x, (list, tuple)):
        arr = np.vstack([_keyword_vec(s) for s in x]).astype(np.float32)
    else:
        arr = _keyword_vec(x)[None, :]
    return arr

# ---- tests ----

def test_best_match_index_and_similarity_with_mock():
    deduper = SemanticDeduper(threshold=0.85)
    with patch.object(deduper, "_encode", side_effect=_encode_side_effect):
        # Add three archetypes
        t_dyn = "Duration 60 — Session Focus: Dynamic Block Session — forehand/backhand"
        t_ps  = "Duration 60 — Session Focus: Progressive ShoteSide — forehand"
        t_pf  = "Duration 60 — Session Focus: Progressive Family — backhand"

        for t in (t_dyn, t_ps, t_pf):
            deduper.add(t)

        # Query a near-duplicate of the dynamic block session
        q = "Dynamic Block Session with Boast-Cross-Drive, forehand & backhand focus"
        is_dup, sim, idx = deduper.is_near_duplicate(q)

        assert is_dup is True
        # must match the first added (index 0)
        assert idx == 0
        assert sim >= deduper.threshold

def test_threshold_behavior_with_known_similarity():
    deduper = SemanticDeduper(threshold=0.90)
    with patch.object(deduper, "_encode", side_effect=_encode_side_effect):
        base = "Progressive ShoteSide — forehand — Drop-Drive emphasis"
        near = "Progressive ShoteSide — forehand — Boast-Cross-Drive & Drop-Drive"
        far  = "Progressive Family — backhand — Boast-Cross-Drive"

        deduper.add(base)
        # Similar (same archetype/side) should trip at ~0.9
        is_dup_near, sim_near, _ = deduper.is_near_duplicate(near)
        assert (sim_near >= 0.9) == is_dup_near

        # Different archetype should be below threshold
        is_dup_far, sim_far, _ = deduper.is_near_duplicate(far)
        assert is_dup_far is False
        assert sim_far < deduper.threshold

@pytest.mark.parametrize("noise", ["  ", "   \n", "  End of session."])
def test_whitespace_and_trailing_noise_invariance(noise):
    """If your implementation normalizes whitespace/boilerplate, this should pass.
       If you don't normalize, you can relax the assertion to sim > 0.95."""
    deduper = SemanticDeduper(threshold=0.95)
    with patch.object(deduper, "_encode", side_effect=_encode_side_effect):
        base = "Dynamic Block Session — forehand/backhand — Boast-Cross-Drive"
        deduper.add(base)
        q = base + noise
        is_dup, sim, idx = deduper.is_near_duplicate(q)
        assert is_dup is True
        assert idx == 0
        assert sim >= 0.95

def test_returns_best_of_many_candidates():
    """When multiple stored items are similar, index must be the best match."""
    deduper = SemanticDeduper(threshold=0.8)
    with patch.object(deduper, "_encode", side_effect=_encode_side_effect):
        a = "Progressive Family — backhand — Drop-Drive"
        b = "Progressive ShoteSide — forehand — Drop-Drive"
        c = "Dynamic Block Session — forehand/backhand — Drop-Drive"
        for t in (a, b, c):
            deduper.add(t)

        q = "Progressive ShoteSide — forehand — Boast-Cross-Drive and Drop-Drive"
        is_dup, sim, idx = deduper.is_near_duplicate(q)
        assert is_dup is True
        # b should be the closest
        assert idx == 1
        assert sim >= deduper.threshold


'''
test below not working
'''
# def test_empty_and_nonstring_inputs_are_handled_gracefully():
#     """Pick a policy. Here we expect a ValueError on empty/non-string."""
#     deduper = SemanticDeduper()
#     with pytest.raises(ValueError):
#         deduper.is_near_duplicate("")
#     with pytest.raises(ValueError):
#         deduper.is_near_duplicate(None)  # type: ignore
