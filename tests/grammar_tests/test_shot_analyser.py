# tests/grammar_tests/test_shot_analyser.py

import pytest
from src.grammar_tools.analysis.shot_analyser import ShotAnalyser

# This test uses a mock of your field retrieval config.
# In a real-world scenario, you might load your actual YAML config here.
MOCK_FIELD_RETRIEVAL_CONFIG = {
    "GENERAL_SHOT_TYPES": {
        "drive": ["drive", "deep drive", "straight drive"],
        "drop": ["drop", "straight drop", "counter drop"],
        "boast": ["boast", "2-wall boast", "3-wall boast"]
    }
}


@pytest.fixture
def analyser() -> ShotAnalyser:
    """Creates a reusable ShotAnalyser instance for all tests."""
    return ShotAnalyser(MOCK_FIELD_RETRIEVAL_CONFIG)


def test_tactical_shot_priority(analyser):
    """
    Tests that a shot explicitly listed in 'tactical_shots' is
    correctly identified as a primary shot.
    """
    session_plan = {
        "blocks": [
            {
                "name": "Activity Block 1",
                "exercises": [
                    (
                        {
                            "family_name": "Drop-Drive",
                            "tactical_shots": ["counter drop"],
                            "foundational_shots": ["deep drive", "straight drop"],
                            "shots": {"specific": ["deep drive", "counter drop", "straight drop"]}
                        }, 9, "points", 7.7
                    )
                ]
            }
        ]
    }

    result = analyser.classify_shots(session_plan)
    assert "counter drop" in result["primary"]
    assert "deep drive" in result["primary"]
    assert "straight drop" in result["primary"]


def test_foundational_shots_are_primary(analyser):
    """
    Tests that shots in 'foundational_shots' are correctly identified
    when no tactical shots are present.
    """
    session_plan = {
        "blocks": [
            {
                "name": "Activity Block 1",
                "exercises": [
                    (
                        {
                            "family_name": "Boast-Cross-Drive",
                            "tactical_shots": [],
                            "foundational_shots": ["2-wall boast", "deep drive"],
                            "shots": {"specific": ["2-wall boast", "deep drive", "straight drive"]}
                        }, 3, "timed", 3.0
                    )
                ]
            }
        ]
    }

    result = analyser.classify_shots(session_plan)
    assert "2-wall boast" in result["primary"]
    assert "deep drive" in result["primary"]
    assert "straight drive" not in result["primary"]

def test_foundational_shots_are_secondary(analyser):
    """
    Tests that shots in 'specific' are correctly identified
    as secondary shots if not in tactical nor in foundational
    """
    session_plan = {
        "blocks": [
            {
                "name": "Activity Block 1",
                "exercises": [
                    (
                        {
                            "family_name": "Boast-Cross-Drive",
                            "tactical_shots": [],
                            "foundational_shots": ["2-wall boast", "deep drive"],
                            "shots": {"specific": ["2-wall boast", "deep drive", "straight drive"]}
                        }, 3, "timed", 3.0
                    )
                ]
            }
        ]
    }

    result = analyser.classify_shots(session_plan)
    assert "straight drive" not in result["primary"]
    assert "straight drive" in result["secondary"]
