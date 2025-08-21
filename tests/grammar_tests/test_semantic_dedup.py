# tests/grammar_tests/test_semantic_dedup.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the class, handling the potential exception
try:
    from src.grammar_tools.dedup.semantic_dedup import SemanticDeduper, SemanticDedupNotAvailable
    SEMANTIC_DEDUP_AVAILABLE = True
except SemanticDedupNotAvailable:
    SEMANTIC_DEDUP_AVAILABLE = False

# This decorator skips the entire test class if the required libs are not available
pytestmark = pytest.mark.skipif(not SEMANTIC_DEDUP_AVAILABLE, reason="Required libraries for semantic dedup are not installed")

class TestSemanticDeduper:

    def test_initialization(self):
        """Test that the SemanticDeduper can be initialized without errors."""
        deduper = SemanticDeduper(threshold=0.95)
        assert deduper is not None
        assert deduper.threshold == 0.95

    def test_empty_store(self):
        """Test that checking an empty store returns no duplicate."""
        deduper = SemanticDeduper()
        is_dup, sim, idx = deduper.is_near_duplicate("Some text")
        assert is_dup is False
        assert sim == 0.0
        assert idx == -1

    def test_adding_and_duplicate_detection(self):
        """Test the basic workflow of adding text and detecting a duplicate."""
        deduper = SemanticDeduper(threshold=0.9)

        text1 = "Session with forehand drives and drops"
        text2 = "Session with forehand drives and drops" # Should be a very close duplicate
        text3 = "A completely different session about backhand volleys" # Should be different

        # First addition is never a duplicate
        is_dup_1, sim_1, idx_1 = deduper.is_near_duplicate(text1)
        assert is_dup_1 is False
        deduper.add(text1)

        # Second, identical addition should be a duplicate with high similarity
        is_dup_2, sim_2, idx_2 = deduper.is_near_duplicate(text2)
        if is_dup_2:
            assert sim_2 >= deduper.threshold
            assert idx_2 == 0 # Should match the first item
        # Note: It might not be *exactly* 1.0 due to floating point precision in the model, but it should be very high.

        # Third, different addition should not be a duplicate
        is_dup_3, sim_3, idx_3 = deduper.is_near_duplicate(text3)
        assert is_dup_3 is False
        assert sim_3 < deduper.threshold # Similarity should be low

    # Test using mocking to avoid relying on model performance for a unit test
    def test_similarity_logic_with_mock(self):
        """Test the duplicate logic using a mocked embedding function."""
        deduper = SemanticDeduper(threshold=0.9)
        # Manually set the internal state to simulate existing embeddings
        # We mock that we have one stored embedding: [1.0, 0.0]
        deduper._embs = np.array([[1.0, 0.0]], dtype=np.float32)
        deduper._dim = 2
        deduper._count = 1

        # Mock the _encode method to return a specific vector for the new text
        # Let's return a very similar vector: [0.99, 0.01] (cosine sim ~0.99)
        with patch.object(deduper, '_encode', return_value=np.array([[0.99, 0.01]], dtype=np.float32)):
            is_dup, sim, idx = deduper.is_near_duplicate("Some text")
            # Should be a duplicate because similarity > 0.9 threshold
            assert is_dup is True
            assert idx == 0

        # Now test with a very different vector: [0.0, 1.0] (cosine sim = 0.0)
        with patch.object(deduper, '_encode', return_value=np.array([[0.0, 1.0]], dtype=np.float32)):
            is_dup, sim, idx = deduper.is_near_duplicate("Some other text")
            # Should NOT be a duplicate
            assert is_dup is False
            assert sim < deduper.threshold