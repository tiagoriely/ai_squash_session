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

def dynamic_query_aware_rrf(
        ranked_lists_map: Dict[str, List[Dict]],
        query: str,
        field_scoring_config: Dict
) -> List[Dict]:
    """
    Hybrid Strategy 3: Fuses lists using a dynamic, query-aware strategy.
    It calculates a "specificity score" based on the query and field weights
    to dynamically adjust the fusion strategy.
    """

    # 1. Parse the query to see which fields the user mentioned
    parsed_desires = parse_user_prompt(query)

    # 2. Calculate the specificity score by summing the base_weights of found fields
    specificity_score = 0.0
    for field_name in parsed_desires.keys():
        # Look up the base_weight for each field in the config
        specificity_score += field_scoring_config.get(field_name, {}).get('base_weight', 0)

    # 3. Define the threshold for what constitutes a "specific" query
    specificity_threshold = 5.5

    # 4. Set Dynamic Weights based on the score
    if specificity_score > specificity_threshold:
        print(f"   -> Query identified as SPECIFIC (Score: {specificity_score:.2f}). Prioritising Field & Sparse.")
        weights = {
            'field_metadata': 1, # 0.5
            'sparse_bm25': 0.0, # 0.3
            'semantic_e5': 0.0 # 0.1
        }
    else:  # Query is VAGUE
        print(f"   -> Query identified as VAGUE (Score: {specificity_score:.2f}). Using balanced, flexible weights.")
        weights = {
            'field_metadata': 0.55,
            'sparse_bm25': 0.15,
            'semantic_e5': 0.3
        }

    # 5. Fuse using the dynamically chosen weights
    return _rrf(ranked_lists_map, weights)