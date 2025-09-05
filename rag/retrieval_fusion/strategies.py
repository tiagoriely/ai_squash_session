# evaluation/retrieval/strategies.py

from collections import defaultdict
from typing import List, Dict

# Import your user query parser to analyse the query
from rag.parsers.user_query_parser import parse_user_prompt


# --- Core Fusion Logic (Helper Function) ---

def _rrf(ranked_lists: Dict[str, List[Dict]], weights: Dict[str, float], k: int = 60) -> List[Dict]:
    """
    Performs weighted Reciprocal Rank Fusion on multiple ranked lists.
    This is the underlying engine for all our hybrid strategies.
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

            # Add to the RRF score, scaled by the retriever's weight
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


# --- HYBRID STRATEGY 1: Static Weights ---

def static_weighted_rrf(ranked_lists_map: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Hybrid Strategy 1: Fuses lists using a fixed set of weights that
    prioritises the high-precision Field Retriever.
    """
    # These weights are static and applied to all queries.
    static_weights = {
        'field_metadata': 0.6,  # Highest weight for the most precise retriever
        'sparse_bm25': 0.25,  # Medium weight for lexical flexibility
        'semantic_e5': 0.15  # Lowest weight, used as a semantic booster
    }

    print("   -> Applying STATIC WEIGHTED RRF strategy.")
    return _rrf(ranked_lists_map, static_weights)


# --- HYBRID STRATEGY 2: Standard Unweighted RRF ---

def standard_unweighted_rrf(ranked_lists_map: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Hybrid Strategy 2: Fuses lists using standard RRF where each
    retriever is treated as an equal expert. This requires no tuning.
    """
    # Equal weights make this a standard, unweighted RRF.
    equal_weights = {
        'field_metadata': 1.0,
        'sparse_bm25': 1.0,
        'semantic_e5': 1.0
    }

    print("   -> Applying STANDARD UNWEIGHTED RRF strategy.")
    return _rrf(ranked_lists_map, equal_weights)


# --- HYBRID STRATEGY 3: Dynamic Query-Aware RRF ---

def dynamic_query_aware_rrf(ranked_lists_map: Dict[str, List[Dict]], query: str) -> List[Dict]:
    """
    Hybrid Strategy 3: Fuses lists using a dynamic, query-aware strategy.
    It analyses the query and adjusts weights to leverage the best retriever.
    """

    # 1. Analyse the Query to determine if it's "vague" or "specific".
    parsed_desires = parse_user_prompt(query)
    specific_keys = {'duration', 'participants', 'squashLevel', 'shots', 'shotSide', 'movement'}
    is_specific = any(key in parsed_desires for key in specific_keys)

    # 2. Set Dynamic Weights based on the query type.
    if is_specific:
        print("   -> Query identified as SPECIFIC. Applying precision-focused weights.")

        weights = {
            'field_metadata': 0.5,  # High weight as it excels at specific queries
            'sparse_bm25': 0.3,  # Complements with lexical matching
            'semantic_e5': 0.2  # Semantic check to ensure relevance
        }
    else:  # Query is VAGUE
        print("   -> Query identified as VAGUE. Applying balanced, flexible weights.")

        weights = {
            'field_metadata': 0.1,  # Low weight as few fields will match
            'sparse_bm25': 0.5,  # Primary signal for finding relevant keyword matches
            'semantic_e5': 0.4  # Strong semantic signal to understand the general intent
        }

    # 3. Fuse using the dynamically chosen weights
    return _rrf(ranked_lists_map, weights)