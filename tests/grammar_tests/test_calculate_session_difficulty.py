# tests/grammar_tests/test_difficulty_calculation.py

import pytest
from src.grammar_tools.analysis.metadata_extractor import MetadataExtractor

@pytest.fixture(scope="module")
def field_retrieval_config():
    """Provides a mock config needed for the extractor to initialise."""
    return {"GENERAL_SHOT_TYPES": {}}

@pytest.fixture
def extractor(field_retrieval_config):
    """Creates a reusable MetadataExtractor instance."""
    return MetadataExtractor(field_retrieval_config)

def _create_mock_exercises(scores: list[int]):
    """Helper function to create mock exercise data from a list of scores."""
    return [({"difficulty_score": s}, None, None, None) for s in scores]

# --- Test Cases ---

@pytest.mark.parametrize("scores, expected_level", [
    ([2, 3, 4], "beginner"),      # Avg <= 4
    ([5, 6, 6], "intermediate"),  # 4 < Avg <= 6
    ([7, 8, 8], "advanced"),      # 6 < Avg <= 8
    ([9, 10, 10], "professional"),# Avg > 8
])
def test_base_level_calculation(extractor, scores, expected_level):
    """Tests the correct base level is calculated from the average score."""
    exercises = _create_mock_exercises(scores)
    result = extractor._calculate_session_difficulty(exercises)
    assert result["recommended"] == expected_level
    assert result["applicable"][0] == expected_level # Check applicable level starts correctly

def test_no_spike_keeps_base_level(extractor):
    """Tests that a session is NOT promoted if spike conditions aren't met."""
    # Avg is 5.5 (Intermediate), but no scores are 7 or higher.
    exercises = _create_mock_exercises([5, 5, 6, 6])
    result = extractor._calculate_session_difficulty(exercises)
    assert result["recommended"] == "intermediate"

def test_applicable_levels_generation(extractor):
    """Tests that the applicable_levels list is generated correctly."""
    # Avg is 5.5, so base level is Intermediate.
    exercises = _create_mock_exercises([5, 6])
    result = extractor._calculate_session_difficulty(exercises)
    assert result["applicable"] == ["intermediate", "advanced", "professional"]

# --- Spike Logic Tests ---

def test_spike_beginner_to_intermediate(extractor):
    """A 'Beginner' session with 2+ exercises >= 5 should be recommended for 'Intermediate'."""
    # Avg is 4.0 (Beginner), but two exercises have a score of 5.
    exercises = _create_mock_exercises([3, 3, 5, 5])
    result = extractor._calculate_session_difficulty(exercises)
    assert result["recommended"] == "intermediate"
    assert result["applicable"][0] == "beginner" # Applicable should still start at the base level

def test_spike_intermediate_to_advanced(extractor):
    """An 'Intermediate' session with 2+ exercises >= 7 should be recommended for 'Advanced'."""
    # Avg is 6.0 (Intermediate), but two exercises have a score of 7.
    exercises = _create_mock_exercises([5, 5, 7, 7])
    result = extractor._calculate_session_difficulty(exercises)
    assert result["recommended"] == "advanced"
    assert result["applicable"][0] == "intermediate"

def test_spike_advanced_to_professional(extractor):
    """An 'Advanced' session with 2+ exercises >= 9 should be recommended for 'Professional'."""
    # Avg is 8.0 (Advanced), but two exercises have a score of 9.
    exercises = _create_mock_exercises([7, 7, 9, 9])
    result = extractor._calculate_session_difficulty(exercises)
    assert result["recommended"] == "professional"
    assert result["applicable"][0] == "advanced"