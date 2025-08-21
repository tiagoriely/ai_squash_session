# tests/grammar_tests/test_metadata_extractor.py

import pytest
from src.grammar_tools.analysis.metadata_extractor import MetadataExtractor

# We need a mock of your field retrieval config for the extractor to initialise.
# In a real test suite, you might load your actual YAML config here.
@pytest.fixture(scope="module")
def field_retrieval_config():
    """Provides a mock field retrieval config for the tests."""
    return {
        "GENERAL_SHOT_TYPES": {
            "drive": ["drive", "deep drive"],
            "drop": ["drop", "counter drop"],
            "boast": ["boast", "2-wall boast"]
        }
    }

@pytest.fixture
def extractor(field_retrieval_config):
    """Creates a reusable MetadataExtractor instance for tests."""
    return MetadataExtractor(field_retrieval_config)

# --- Test Data ---

@pytest.fixture
def drill_only_session():
    """A mock session plan containing only drills."""
    return {
        "blocks": [{
            "name": "Activity Block 1",
            "exercises": [
                (
                    {"types": ["drill"], "movement": ["diagonal"]},
                    3, "timed", 3.0
                )
            ]
        }],
        "meta": {"archetype": "Test Archetype", "duration": 45}
    }

@pytest.fixture
def conditioned_game_session():
    """A mock session plan containing only conditioned games."""
    return {
        "blocks": [{
            "name": "Activity Block 1",
            "exercises": [
                (
                    {"types": ["conditioned_game"], "movement": ["front"]},
                    9, "points", 7.7
                )
            ]
        }],
        "meta": {}
    }

@pytest.fixture
def mixed_session():
    """A mock session containing both drills and conditioned games."""
    return {
        "blocks": [
            {"name": "Warm-up", "exercises": [({"types": ["warmup"], "movement": ["sideways"]}, 3, "timed", 3.0)]},
            {
                "name": "Activity Block 1",
                "exercises": [
                    ({"types": ["drill"], "movement": ["diagonal", "front"]}, 3, "timed", 3.0),
                    ({"types": ["conditioned_game"], "movement": ["diagonal", "back"]}, 9, "points", 7.7)
                ]
            }
        ],
        "meta": {"context": {"must_use_side": "forehand"}}
    }

@pytest.fixture
def full_conditioned_game_session():
    """
    A mock session plan where ALL activity exercises are conditioned games,
    which should result in a 'conditioned_game' session_type.
    """
    return {
        "blocks": [
            {"name": "Warm-up", "exercises": [({"types": ["warmup"]}, 3, "timed", 3.0)]},
            {
                "name": "Activity Block 1",
                "exercises": [
                    ({"types": ["conditioned_game"]}, 9, "points", 7.7),
                    ({"types": ["conditioned_game"]}, 11, "points", 9.4)
                ]
            },
            {
                "name": "Activity Block 2",
                "exercises": [
                    ({"types": ["conditioned_game"]}, 7, "points", 6.0)
                ]
            }
        ],
        "meta": {}
    }

@pytest.fixture
def full_mixed_session():
    """
    A mock session with a mix of drills and games, which should
    result in a 'mix' session_type.
    """
    return {
        "blocks": [
            {"name": "Warm-up", "exercises": [({"types": ["warmup"]}, 3, "timed", 3.0)]},
            {
                "name": "Activity Block 1",
                "exercises": [
                    ({"types": ["drill"]}, 3, "timed", 3.0),
                    ({"types": ["conditioned_game"]}, 11, "points", 9.4)
                ]
            },
            {
                "name": "Activity Block 2",
                "exercises": [
                    ({"types": ["drill"]}, 4, "timed", 4.0)
                ]
            }
        ],
        "meta": {}
    }

# --- Tests ---

def test_session_type_drill(extractor, drill_only_session):
    """It should identify a session with only drills as type 'drill'."""
    metadata = extractor.generate_rag_metadata(drill_only_session)
    assert metadata["session_type"] == "drill"

def test_session_type_conditioned_game(extractor, conditioned_game_session):
    """It should identify a session with only conditioned games as type 'conditioned_game'."""
    metadata = extractor.generate_rag_metadata(conditioned_game_session)
    assert metadata["session_type"] == "conditioned_game"

def test_session_type_mix(extractor, mixed_session):
    """It should correctly identify a mixed session."""
    metadata = extractor.generate_rag_metadata(mixed_session)
    assert metadata["session_type"] == "mix(conditioned_game, drill)" # Corrected assertion

def test_full_session_is_correctly_identified_as_conditioned_game(extractor, full_conditioned_game_session):
    """
    Given a session with multiple blocks containing only conditioned games,
    it should correctly identify the overall session type.
    """
    metadata = extractor.generate_rag_metadata(full_conditioned_game_session)
    assert metadata["session_type"] == "conditioned_game"

def test_full_session_is_correctly_identified_as_mix(extractor, full_mixed_session):
    """
    Given a session with multiple blocks containing a mix of drills and games,
    it should correctly identify the overall session type as a mix.
    """
    metadata = extractor.generate_rag_metadata(full_mixed_session)
    assert metadata["session_type"] == "mix(conditioned_game, drill)" # Corrected assertion


def test_movement_aggregation(extractor, mixed_session):
    """It should aggregate unique movement types from all activity exercises."""
    metadata = extractor.generate_rag_metadata(mixed_session)
    # Note: 'sideways' from the warm-up should be excluded.
    assert metadata["movement"] == ["back", "diagonal", "front"]

def test_shot_side_single(extractor, mixed_session):
    """It should correctly extract a single shot side from the context."""
    metadata = extractor.generate_rag_metadata(mixed_session)
    assert metadata["shotSide"] == "forehand"

def test_shot_side_default(extractor, drill_only_session):
    """It should default to both sides if no specific side is required."""
    metadata = extractor.generate_rag_metadata(drill_only_session)
    assert metadata["shotSide"] == ["forehand", "backhand"]

def test_participants_default(extractor, drill_only_session):
    """It should correctly default the number of participants."""
    metadata = extractor.generate_rag_metadata(drill_only_session)
    assert metadata["participants"] == 2