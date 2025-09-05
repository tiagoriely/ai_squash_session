# tests/test_field_retriever.py
import sys
from pathlib import Path
from collections.abc import Mapping

import pytest

# Ensure package imports work regardless of how pytest is invoked.
# Adjust this if your test file lives elsewhere.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rag.retrieval.field_retriever import FieldRetriever
from rag.parsers.user_query_parser import parse_user_prompt


# --- Test Data (kept minimal and deterministic) ---
@pytest.fixture(scope="module")
def mock_kb():
    return [
        {
            "session_id": "doc_1",
            "type": "conditioned_game",
            "duration": 45,
            "squashLevel": "advanced",
            "contents": "A 45-minute session of conditioned games for advanced players.",
        },
        {
            "session_id": "doc_2",
            "type": "drill",
            "duration": 30,
            "squashLevel": "intermediate",
            "contents": "A 30-minute drill session.",
        },
    ]


@pytest.fixture(scope="module")
def config_path():
    # Path to your retrieval config, relative to the project root.
    cfg = project_root / "configs" / "retrieval" / "raw_squash_field_retrieval_config.yaml"
    if not cfg.exists():
        pytest.skip(f"Missing config file: {cfg}")
    return cfg


@pytest.fixture(scope="module")
def retriever(mock_kb, config_path):
    # Initialises once for all tests in this module.
    return FieldRetriever(knowledge_base=mock_kb, config_path=config_path)


# --- Parser behaviour ---
@pytest.mark.parametrize(
    "query",
    [
        "a drill for squash",
        "a 45-minute conditioned game session",
    ],
)
def test_parse_user_prompt_returns_mapping(query):
    """Parser should extract a structured mapping from free-text queries."""
    user_desires = parse_user_prompt(query)
    assert user_desires, "Parser returned empty/falsey output."
    assert isinstance(user_desires, Mapping), "Parser should return a mapping-like structure."


# --- Retriever behaviour ---
@pytest.mark.parametrize(
    ("query", "expected_top_id"),
    [
        ("a drill for squash", "doc_2"),
        ("a 45-minute conditioned game session", "doc_1"),
    ],
)
def test_retriever_returns_expected_top_document(retriever, query, expected_top_id):
    """Retriever should surface the most relevant document for simple, unambiguous queries."""
    results = retriever.search(query, top_k=1)

    # Basic shape checks
    assert isinstance(results, list), "Retriever should return a list."
    assert results, "Retriever returned no results."

    top = results[0]
    assert top.get("session_id") == expected_top_id, "Unexpected top result."
    assert "field_score" in top, "Top result should include a final/field score."


def test_retriever_handles_no_match_gracefully(retriever):
    """For queries with no obvious match, retriever should not error."""
    results = retriever.search("completely unrelated query about astronomy", top_k=1)
    # Depending on your implementation, this may be [] or a low-scoring item.
    # We only assert it doesn't error and returns a list.
    assert isinstance(results, list)
