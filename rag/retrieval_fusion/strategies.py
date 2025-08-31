from collections import defaultdict
from typing import List, Dict

# Import your user query parser to analyze the query
from rag.parsers.user_query_parser import parse_user_prompt


# --- Reciprocal Rank Fusion (RRF) Helper ---
def _rrf(ranked_lists: Dict[str, List[Dict]], weights: Dict[str, float], k: int = 60) -> List[Dict]:
    """
    Performs weighted Reciprocal Rank Fusion on multiple ranked lists.

    Args:
        ranked_lists (Dict[str, List[Dict]]): A dictionary mapping retriever names to their ranked lists.
        weights (Dict[str, float]): A dictionary mapping retriever names to their fusion weights.
        k (int): A constant to control the influence of lower-ranked documents.

    Returns:
        List[Dict]: The final, fused, and re-ranked list of documents.
    """
    rrf_scores = defaultdict(float)
    doc_inventory = {}

    for retriever_name, docs in ranked_lists.items():
        weight = weights.get(retriever_name, 0.0)
        if not docs or weight == 0:
            continue

        for rank, doc in enumerate(docs):
            doc_id = doc.get('id') or doc.get('session_id')
            if not doc_id:
                continue

            # Add to the RRF score
            rrf_scores[doc_id] += weight * (1 / (k + rank + 1))

            # Keep the document content from its first appearance
            if doc_id not in doc_inventory:
                doc_inventory[doc_id] = doc

    # Sort documents by their final RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)

    # Create the final ranked list
    final_ranked_list = []
    for doc_id in sorted_doc_ids:
        doc = doc_inventory[doc_id]
        doc['fusion_score'] = rrf_scores[doc_id]
        final_ranked_list.append(doc)

    return final_ranked_list


# --- The Main Query-Aware Fusion Strategy ---
def query_aware_fusion(ranked_lists_map: Dict[str, List[Dict]], query: str) -> List[Dict]:
    """
    Fuses ranked lists using a dynamic, query-aware strategy.

    1. Applies reliability thresholds to filter results.
    2. Analyzes the query to determine if it's "vague" or "specific".
    3. Sets dynamic weights for RRF based on the query type.
    4. Fuses the filtered lists using weighted RRF.
    """

    # 1. Apply Reliability Thresholds
    thresholded_lists = {}
    for retriever_name, docs in ranked_lists_map.items():
        if 'semantic' in retriever_name:
            thresholded_lists[retriever_name] = [doc for doc in docs if doc.get('semantic_score', 0) >= 0.70]
        elif 'field' in retriever_name:
            thresholded_lists[retriever_name] = [doc for doc in docs if doc.get('field_score', 0) >= 3.0]
        else:
            # No threshold for the sparse retriever as it's already robust
            thresholded_lists[retriever_name] = docs

    # 2. Analyze the Query
    # (Note: Assumes your parser can be called without the config/durations for this check)
    parsed_desires = parse_user_prompt(query)
    # Define "specific" as a query where key metadata is extracted
    specific_keys = {'duration', 'participants', 'squashLevel', 'shots'}
    is_specific = any(key in parsed_desires for key in specific_keys)

    # 3. Set Dynamic Weights
    if is_specific:
        print("   -> Query identified as SPECIFIC. Prioritizing sparse and field retrievers.")
        weights = {
            'semantic_e5': 0.1,
            'sparse_bm25': 0.45,
            'field_metadata': 0.45
        }
    else:
        print("   -> Query identified as VAGUE. Prioritizing semantic retriever.")
        weights = {
            'semantic_e5': 0.6,
            'sparse_bm25': 0.2,
            'field_metadata': 0.2
        }

    # 4. Fuse using weighted RRF
    fused_results = _rrf(thresholded_lists, weights)

    return fused_results