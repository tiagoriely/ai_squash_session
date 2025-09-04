# tests/retrieval/test_field_retriever.py

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from rag.retrieval.field_retriever import FieldRetriever
from rag.parsers.user_query_parser import parse_user_prompt
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter

# --- Test Setup & Fixtures ---

# Define the absolute path to your project's root directory
# Adjust this path if your test runner's working directory is different.
PROJECT_ROOT = Path("/Users/tiago/projects/ai_squash_session")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Helper function to load a .jsonl file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def raw_corpus() -> List[Dict[str, Any]]:
    """Loads the raw 10-document test corpus."""
    corpus_path = PROJECT_ROOT / "data/processed/balanced_grammar/balanced_10.jsonl"
    assert corpus_path.exists(), f"Test corpus not found at {corpus_path}"
    return load_jsonl(corpus_path)

@pytest.fixture(scope="module")
def adapted_corpus(raw_corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transforms the raw corpus into a flat structure using the adapter."""
    adapter = SquashNewCorpusAdapter()
    return [adapter.transform(doc) for doc in raw_corpus]


@pytest.fixture(scope="module")
def field_retriever(adapted_corpus: List[Dict[str, Any]]) -> FieldRetriever:
    """Initialises the FieldRetriever with the *adapted* test corpus."""
    config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    assert config_path.exists(), f"FieldRetriever config not found at {config_path}"
    # The retriever's knowledge base is now the adapted, flat-structured data
    return FieldRetriever(knowledge_base=adapted_corpus, config_path=config_path)


# --- Test Cases ---

@pytest.mark.parametrize(
    "query, target_doc_id, expected_score",
    [
        (
                "I want a 45-minute conditioned game for two intermediate players",
                "session_004",
                14.5  # type(5) + level(4) + duration(3) + participants(2.5) = 14.5
        ),
        (
                "An advanced drill session for about 45 minutes",
                "session_005",
                12.0  # type(5) + level(4) + duration(3) = 12.0
        ),
        (
                "A 60 minute session with a mix of drills and games, focusing on backhand",
                "session_006",
                9.5  # type(5) + duration(3) + shotSide(1.5) = 9.5
        ),
        # --- From Under-specified set ---
        # This query has one specific desire ('drill'), so it should score.
        (
                "a drill for squash",
                "session_005",
                5.0  # Matches only on type (5.0)
        ),

        # --- From Out-of-distribution set (but valid) ---
        # This query only specifies shots. The score is based on the complex hierarchical scoring.
        (
                "a session focusing on the volley cross",
                "session_002",
                2.357142857142857
        # Score comes entirely from shot overlap, which is complex. Most docs with full general shots get this.
        ),

        # --- From Graduated Complexity set ---
        # We expect the score to progressively increase as the query gets more specific.
        (
                "a conditioned game session",
                "session_004",
                5.0  # Score: type (5.0)
        ),
        (
                "a 45-minute conditioned game session",
                "session_004",
                8.0  # Score: type (5.0) + duration (3.0)
        ),
        (
                "a 45-minute conditioned game session for an advanced player",
                "session_004",
                11.0  # Score: 8.0 + applicable_squash_level (4.0 * 0.75)
        ),
        (
                "a 45-minute conditioned game session for an advanced player focusing on volley drops",
                "session_004",
                13.357142857142858  # Score: 11.0 + shots score (~2.36)
        ),
    ]
)
def test_individual_document_scoring(
        field_retriever: FieldRetriever,
        adapted_corpus: List[Dict[str, Any]],
        query: str,
        target_doc_id: str,
        expected_score: float
):
    """
    Unit test for the scoring logic on a single, specific document.
    This is great for precise debugging and validation.
    """
    # Find the specific document we want to score
    target_doc = next((doc for doc in adapted_corpus if doc["id"] == target_doc_id), None)
    assert target_doc is not None, f"Document with id '{target_doc_id}' not found in corpus."

    # Parse the user query into desired fields
    user_desires = parse_user_prompt(query)
    print(f" User desire found: {user_desires}")

    # Calculate the score for only that document
    actual_score = field_retriever._score_document(target_doc, user_desires)

    print(
        f"\n[Scoring Test] Query: '{query}' on Doc: '{target_doc_id}' -> Expected: {expected_score}, Got: {actual_score:.2f}")

    # Assert that the calculated score is what we expect
    assert actual_score == pytest.approx(expected_score)


# --- Test Case: Top-K Ranking ---

@pytest.mark.parametrize(
    "query, expected_top_ids, min_expected_score",
    [
        # --- Original Queries ---
        ("I want a 45-minute conditioned game for two intermediate players", ["session_004"], 14.0),
        ("An advanced drill session for about 45 minutes", ["session_005", "session_009"], 11.0),
        ("A 60 minute session with a mix of drills and games, focusing on backhand", ["session_006"], 9.0),

        # --- Under-specified but valid query ---
        ("a drill for squash", ["session_005", "session_009"], 4.9),  # Should just match on type (score 5.0)

        # --- Out-of-distribution but valid query ---
        ("a session focusing on the volley cross", ["session_005"], 2.39),
        # Matches on shots, complex scoring

        # --- Graduated Complexity Queries ---
        # Score should increase as query becomes more specific
        ("a conditioned game session", ["session_004", "session_007"], 4.9),  # Score: 5.0 (type)
        ("a 45-minute conditioned game session", ["session_004"], 7.9),  # Score: 5.0 (type) + 3.0 (duration) = 8.0
        ("a 45-minute conditioned game session for an advanced player", ["session_004"], 10.9),
        # Score: 8.0 + 3.0 (applicable level) = 11.0
        ("a 45-minute conditioned game session for an advanced player focusing on volley drops", ["session_004"],
         11.0),  # Score will be > 11.0
    ]
)
def test_high_score_queries(
        field_retriever: FieldRetriever,
        query: str,
        expected_top_ids: list[str],
        min_expected_score: float
):
    """
    Tests queries that are expected to return a high score and rank a specific document at the top.
    Handles cases where multiple documents can share the top score.
    """
    results = field_retriever.search(query, top_k=3)

    assert results, f"Query '{query}' returned no results, but expected a match."
    top_doc = results[0]

    # CHANGED: Assert that the top document's ID is in the list of expected IDs
    top_doc_id = top_doc['id']
    assert top_doc_id in expected_top_ids, \
        f"Query '{query}' returned '{top_doc_id}' as top result, but expected one of {expected_top_ids}."

    top_score = top_doc['field_score']
    print(f"\n[Ranking Test] Query: '{query}' -> Top Doc ID: {top_doc_id}, Score: {top_score:.2f}")
    assert top_score >= min_expected_score

    scores = [res['field_score'] for res in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted."


@pytest.mark.parametrize(
    "query, max_expected_score",
    [
        # Query 1: On-topic but attributes don't match the small corpus well
        ("A 20-minute solo drill for a beginner", 5.0),  # Low duration, solo=1 participant, beginner level
        # Query 2: Asks for shots not prominent in the top documents
        ("Show me ghosting routines that focus on the boast shot", 2.25),
        # Type 'ghosting' is a match, but 'nick' is not a primary shot in sample docs
        ("a solo session to practice cross drops", 4.0)
    ]
)


def test_poor_score_queries(field_retriever: FieldRetriever, query: str, max_expected_score: float):
    """
    Tests queries that are on-topic but should not score highly against the corpus.

    ASSERTIONS:
    1.  If results are returned, the top score must be below a specified maximum threshold.
    """
    results = field_retriever.search(query, top_k=1)

    if not results:
        # Returning no results is a valid outcome for a poor query.
        print(f"\n[Poor-Score Test] Query: '{query}' -> Correctly returned no results.")
        assert True
    else:
        top_score = results[0]['field_score']
        print(f"\n[Poor-Score Test] Query: '{query}' -> Top Score: {top_score:.2f}")
        assert top_score <= max_expected_score, \
            f"Top score for poor query '{query}' was {top_score:.2f}, which is higher than the max expected score of {max_expected_score:.2f}."


@pytest.mark.parametrize(
    "query",
    [
        "What are the best tennis drills for improving my serve?",
        "I need a good badminton practice plan",
        "Show me some padel exercises",
        "a drill to improve my tennis serve",
        "how to practice a badminton drop shot",
        "a good warm-up for playing padel"
    ]
)
def test_irrelevant_queries_return_nothing(field_retriever: FieldRetriever, query: str):
    """
    Tests queries containing negative keywords (e.g., other sports).

    ASSERTION:
    1. The retriever must return an empty list for these queries.
    """
    results = field_retriever.search(query, top_k=3)
    print(f"\n[Irrelevant-Query Test] Query: '{query}' -> Results returned: {len(results)}")
    assert results == [], \
        f"Query '{query}' should be irrelevant and return an empty list, but it returned {len(results)} results."