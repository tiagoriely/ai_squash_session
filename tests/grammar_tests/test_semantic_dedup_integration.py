# tests/grammar_tests/test_semantic_dedup_integration.py
import numpy as np
import pytest

try:
    from src.grammar_tools.dedup.semantic_dedup import SemanticDeduper, SemanticDedupNotAvailable
    SEMANTIC_DEDUP_AVAILABLE = True
except Exception:
    SEMANTIC_DEDUP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SEMANTIC_DEDUP_AVAILABLE, reason="Semantic libs not installed")

def test_identical_text_is_self_duplicate_with_high_similarity():
    deduper = SemanticDeduper(threshold=0.99)
    text = "Dynamic Block Session — forehand/backhand — Boast-Cross-Drive."
    deduper.add(text)
    is_dup, sim, idx = deduper.is_near_duplicate(text)
    # identical strings should be a perfect (or near-perfect) match
    assert is_dup is True
    assert idx == 0
    assert sim == pytest.approx(1.0, rel=1e-5, abs=1e-5)

@pytest.mark.slow
def test_large_store_still_finds_right_item():
    """Optionally exercise ANN/search path if you have one."""
    deduper = SemanticDeduper(threshold=0.85)
    base = "Progressive ShoteSide — forehand — Drop-Drive emphasis."
    deduper.add(base)
    # Add distractors
    for i in range(250):
        deduper.add(f"Filler item {i}: Progressive Family — backhand — lob/kill.")
    is_dup, sim, idx = deduper.is_near_duplicate("Progressive ShoteSide forehand Drop-Drive.")
    assert is_dup is True
    assert sim >= 0.85
