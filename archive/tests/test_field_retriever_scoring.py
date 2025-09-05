# tests/test_field_retriever_scoring.py
import sys
from pathlib import Path
import pytest

# Add the project root to the Python path to allow for package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.retrieval.field_retriever import FieldRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter

# --- Test Data ---
# Two documents that are identical, except for how the squash level is defined.
RAW_KNOWLEDGE_BASE = [
    {
        "id": "doc_recommended",
        "meta": {
            "session_type": "drill",
            "duration": 30,
            # 'advanced' is the recommended level
            "recommended_squash_level": "advanced",
            "applicable_squash_levels": ["intermediate", "advanced", "professional"]
        },
        "contents": "A test drill session."
    },
    {
        "id": "doc_applicable",
        "meta": {
            "session_type": "drill",
            "duration": 30,
            # 'advanced' is only an applicable level, not recommended
            "recommended_squash_level": "intermediate",
            "applicable_squash_levels": ["intermediate", "advanced", "professional"]
        },
        "contents": "A test drill session."
    }
]

@pytest.fixture
def retriever() -> FieldRetriever:
    """A pytest fixture to set up the retriever for the test."""
    # 1. Adapt the raw data into the flat format the retriever expects.
    adapter = SquashNewCorpusAdapter()
    adapted_kb = [adapter.transform(doc) for doc in RAW_KNOWLEDGE_BASE]

    # 2. Set up the retriever.
    config_path = project_root / "configs" / "retrieval" / "raw_squash_field_retrieval_config.yaml"
    return FieldRetriever(knowledge_base=adapted_kb, config_path=config_path)


def test_squash_level_scoring_priority(retriever: FieldRetriever):
    """
    Tests that a document matching the 'recommended_squash_level' gets a higher score
    and is ranked first compared to one matching only in 'applicable_squash_levels'.
    """
    # Arrange: The query only asks for a level, not a type.
    query = "a session for an advanced player"

    # Act: Run the search. We ask for top_k=2 to get both results.
    results = retriever.search(query, top_k=2)

    # Assert: Verify the scores and ranking.
    assert len(results) == 2, "Should have found both documents"

    # Separate the results for easier checking
    result_recommended = next((r for r in results if r['id'] == 'doc_recommended'), None)
    result_applicable = next((r for r in results if r['id'] == 'doc_applicable'), None)

    assert result_recommended is not None, "Recommended document should be in the results"
    assert result_applicable is not None, "Applicable document should be in the results"

    # 1. Check Ranking: The recommended document should be the first result.
    assert results[0]['id'] == 'doc_recommended', "The 'recommended' match should be ranked higher"

    # 2. Check Score Difference: The recommended score should be greater than the applicable score.
    score_recommended = result_recommended['field_score']
    score_applicable = result_applicable['field_score']

    print(f"Score (Recommended): {score_recommended}, Score (Applicable): {score_applicable}")
    assert score_recommended > score_applicable, "The 'recommended' score should be higher"

    # --- FIX: Update the asserted scores ---
    # 3. Check exact scores based on your config (squashLevel base_weight: 3.0)
    # The query ONLY matches the squashLevel, so the 'type' score is not added.
    # Recommended gets 100% of the weight: 3.0
    # Applicable gets 90% of the weight: 3.0 * 0.9 = 2.7
    assert score_recommended == pytest.approx(3.0)
    assert score_applicable == pytest.approx(2.7)