# rag/fusion/methods.py
from typing import List, Dict
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Performs Reciprocal Rank Fusion on multiple lists of ranked documents.
    Logic from hybrid_retriever_rrf.py
    """
    rrf_scores = defaultdict(float)
    doc_lookup = {}

    # Process each list of ranked documents
    for ranked_list in ranked_lists:
        for i, doc in enumerate(ranked_list):
            doc_id = doc['id']
            rank = i + 1
            # Add to the RRF score
            rrf_scores[doc_id] += 1 / (k + rank)
            # Store the document itself for later retrieval
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

    # Sort the document IDs by their final RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)

    # Build the final list of results
    final_results = []
    for doc_id in sorted_ids:
        doc = doc_lookup[doc_id]
        doc['rrf_score'] = rrf_scores[doc_id]
        final_results.append(doc)

    return final_results


def rerank_by_weighted_score(candidates: List[Dict], alpha: float = 0.7) -> List[Dict]:
    """
    Re-ranks candidates by combining normalized field scores and semantic scores with a weighted average.
    Logic from hybrid_retriever_semantic_then_field.py
    """
    if not candidates:
        return []

    # Find the maximum field score for normalization
    max_field_score = max(c.get('field_score', 0) for c in candidates)

    for c in candidates:
        # Normalize the field score to a 0-1 range
        norm_field_score = (c.get('field_score', 0) / max_field_score) if max_field_score > 0 else 0.0
        semantic_score = c.get('semantic_score', 0)
        # Calculate the final combined score
        c['final_score'] = (alpha * norm_field_score) + ((1 - alpha) * semantic_score)

    # Sort the candidates by the new final_score
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates


def sort_by_field_then_semantic(candidates: List[Dict]) -> List[Dict]:
    """
    Sorts candidates first by 'field_score', then by 'semantic_score' as a tie-breaker.
    Logic from hybrid_retriever_field_then_semantic.py
    """
    # sort by semantic score first, then field score.
    candidates.sort(key=lambda x: (x.get('field_score', 0), x.get('semantic_score', 0)), reverse=True)
    return candidates