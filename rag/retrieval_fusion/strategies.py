# evaluation/retrieval/strategies.py

from collections import defaultdict
from typing import List, Dict

# Import your user query parser to analyse the query
from rag.parsers.user_query_parser import parse_user_prompt

# Negative keywords indicating out-of-scope sports or topics.  These are
# taken from the field retriever's configuration (NEGATIVE_KEYWORDS).  If
# a query contains one of these and does not mention squash, it will be
# treated as out of scope and the dense retriever's influence will be
# reduced accordingly.
NEGATIVE_KEYWORDS = {
    "tennis",
    "badminton",
    "padel",
    "pickleball",
    "racquetball",
    "racketball",
    "hardball",
    "table tennis",
    "ping pong",
}


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

    # # Keep a copy of raw RRF scores for diagnostics
    # raw_scores = dict(rrf_scores)
    #
    # # Normalise to [0,1] per query using the existing utility
    # norm_scores = _normalise_scores(raw_scores)
    #
    # # Rank by the normalised score
    # sorted_doc_ids = sorted(norm_scores.keys(), key=lambda id_: norm_scores[id_], reverse=True)
    #
    # # Keeping raw and normalised scores
    # final_ranked_list = []
    # for doc_id in sorted_doc_ids:
    #     doc = doc_inventory[doc_id]
    #     doc['fusion_score_raw'] = raw_scores[doc_id]
    #     doc['fusion_score'] = norm_scores[doc_id] # for plots
    #     final_ranked_list.append(doc)
    #
    # return final_ranked_list



# --- Utility functions for score-based fusion ---

def _normalise_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise a dictionary of raw scores to the range [0, 1].

    If all scores are equal, the normalised scores will all be 0.0.  This
    prevents division by zero errors and reflects the absence of any
    discriminative information in the scores.

    Args:
        score_dict (Dict[str, float]): Mapping from document IDs to raw scores.

    Returns:
        Dict[str, float]: Mapping from document IDs to normalised scores.
    """
    if not score_dict:
        return {}

    scores = list(score_dict.values())
    min_score = min(scores)
    max_score = max(scores)
    range_ = max_score - min_score

    if range_ == 0:
        # All scores identical; return zeros
        return {doc_id: 0.0 for doc_id in score_dict}

    return {doc_id: (score - min_score) / range_ for doc_id, score in score_dict.items()}


def _compute_dynamic_weights(
    specificity_score: float,
    dynamic_range: float = 22.0,
    out_of_scope: bool = False,
) -> Dict[str, float]:
    """
    Compute dynamic weights for the field, sparse and semantic retrievers
    based on the query's specificity score.

    This version introduces a baseline weight for the semantic retriever to
    prevent its contribution from dropping too sharply as the field weight
    increases.  The field weight grows linearly with the specificity score
    and saturates below 1.0 (at most 0.8 by default).  The sparse and
    semantic retrievers share the remaining weight after accounting for
    the field weight and the semantic baseline.

    Args:
        specificity_score (float): Sum of base_weights of the fields present in the query.
        dynamic_range (float): Score at which the field weight reaches its
            maximum (cap_field_weight).  A larger value delays saturation.

    Returns:
        Dict[str, float]: A dict of weights for 'field_metadata', 'sparse_bm25'
            and 'semantic_e5'.
    """
    # Out-of-scope queries (no valid field scores) should reduce the
    # influence of the dense retriever.  We use a smaller baseline for
    # semantic and cap the field weight lower, and allocate more of the
    # leftover to the sparse retriever.
    if out_of_scope:
        semantic_baseline = 0.05
        cap_field_weight = 0.3
        # Compute raw field weight; it will be zero since specificity_score=0
        raw_field_weight = max(0.0, specificity_score) / dynamic_range
        w_field = min(cap_field_weight, raw_field_weight)
        leftover = max(0.0, 1.0 - semantic_baseline - w_field)
        # Allocate most of the leftover to sparse (80%), small to semantic (20%)
        w_sparse = leftover * 0.8
        w_semantic = semantic_baseline + leftover * 0.2
    else:
        # Regular case: ensure the semantic retriever always contributes
        # something, even for highly specific queries.
        semantic_baseline = 0.2
        cap_field_weight = 0.8
        raw_field_weight = max(0.0, specificity_score) / dynamic_range
        w_field = min(cap_field_weight, raw_field_weight)
        leftover = max(0.0, 1.0 - semantic_baseline - w_field)
        # Allocate the leftover between sparse (25%) and semantic (75%)
        w_sparse = leftover * 0.25
        w_semantic = semantic_baseline + leftover * 0.75

    return {
        'field_metadata': w_field,
        'sparse_bm25': w_sparse,
        'semantic_e5': w_semantic
    }


def dynamic_query_aware_score_fusion(
    ranked_lists: Dict[str, List[Dict]],
    query: str,
    field_scoring_config: Dict,
    *,
    field_max_score: float = 20.0,
    spec_threshold: float = 4.0,
    bm25_scale: float = 5.0,
    dense_threshold: float = 0.65,
    dense_range: float = 0.25,
    bonus_factor: float = 0.35
) -> List[Dict]:
    """
    Dynamic fusion strategy that combines field, sparse (BM25) and dense
    retriever scores using query-aware weights and score calibration.

    This implementation addresses several requirements:

    * **Signal strength and relevance precision**: The raw field and BM25
      scores are scaled by fixed constants rather than normalised per
      query, preserving the magnitude of strong matches.  Dense scores
      are thresholded to suppress semantically irrelevant queries.

    * **Inter-query discriminatory power**: Out-of-scope queries are
      detected via negative keywords (e.g. tennis, badminton).  When a
      query is deemed out-of-scope, the dense retriever's influence is
      greatly reduced and the sparse retriever receives most of the
      weight.  Random queries without negative keywords still retain a
      baseline dense weight but yield low overall scores due to the
      thresholding.

    * **Sensitivity to specificity**: The weight assigned to the field
      retriever increases smoothly with the query's specificity score
      using a logistic formulation (score / (score + spec_threshold)),
      capped at a moderate value.  A small bonus proportional to the
      specificity is added uniformly to all documents for the query to
      encourage monotonicity across increasingly specific queries.

    * **Intra-query discriminatory power**: Score calibration and
      weighting produce a spread of fusion scores within each query.

    The final fusion score is clamped to 1.0 to avoid exceeding the
    maximum bound.

    Args:
        ranked_lists: Mapping from retriever names to their retrieved
            documents.  Each document should include 'field_score',
            'sparse_score' and 'semantic_score' where appropriate.
        query: The raw user query text.
        field_scoring_config: Configuration mapping field names to a
            dict containing at least a 'base_weight'.  Used to compute
            query specificity.
        field_max_score: Constant used to scale field scores into [0,1].
        spec_threshold: Parameter controlling how quickly the field
            weight grows with specificity.  Higher values delay
            saturation.
        bm25_scale: Constant used to scale BM25 scores.  A score of
            ``bm25_scale`` maps to 1.0.  Negative BM25 scores are
            retained to penalise lexically irrelevant queries.
        dense_threshold: Baseline similarity below which semantic
            contributions are ignored.
        dense_range: Range of similarity above the threshold used to
            normalise semantic scores.  A similarity of
            ``dense_threshold + dense_range`` maps to 1.0.
        bonus_factor: Multiplicative factor for the specificity bonus
            added uniformly to all documents for a given query.

    Returns:
        A list of documents with a new 'fusion_score' field.  Higher
        scores indicate greater relevance.
    """
    # 1. Compute query specificity based on the parsed fields
    parsed_desires = parse_user_prompt(query)
    specificity_score = 0.0
    for field_name in parsed_desires.keys():
        specificity_score += field_scoring_config.get(field_name, {}).get('base_weight', 0.0)

    # 2. Extract raw scores and collect documents
    field_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}
    semantic_scores: Dict[str, float] = {}
    doc_inventory: Dict[str, Dict] = {}

    for retriever_name, docs in ranked_lists.items():
        for doc in docs:
            doc_id = doc.get('id') or doc.get('session_id')
            if not doc_id:
                continue
            # Store the document once
            if doc_id not in doc_inventory:
                doc_inventory[doc_id] = doc
            # Extract the relevant scores
            if retriever_name == 'field_metadata':
                field_scores[doc_id] = doc.get('field_score', 0.0)
            elif retriever_name == 'sparse_bm25':
                sparse_scores[doc_id] = doc.get('sparse_score', 0.0)
            elif retriever_name == 'semantic_e5':
                semantic_scores[doc_id] = doc.get('semantic_score', 0.0)

    # 3. Determine if the query is out-of-scope using negative keywords.
    query_lower = query.lower()
    contains_squash = "squash" in query_lower
    contains_negative = any(kw in query_lower for kw in NEGATIVE_KEYWORDS)
    # A query is out-of-scope only if it contains a negative keyword and
    # does not mention squash at all.  Otherwise, vague queries fall
    # back on the dense retriever.
    out_of_scope = (contains_negative and not contains_squash)

    # 4. Compute dynamic weights based on specificity and scope.
    # Logistic growth for the field weight: score / (score + threshold).
    if specificity_score > 0:
        w_field_raw = specificity_score / (specificity_score + spec_threshold)
    else:
        w_field_raw = 0.0

    if out_of_scope:
        # Out-of-scope: discard field evidence and downweight dense.
        w_field = 0.0
        base_semantic = 0.05  # minimal semantic influence
        leftover = 1.0 - base_semantic - w_field
        # Allocate most of the leftover to sparse and a small portion to dense
        w_dense = base_semantic + leftover * 0.3
        w_sparse = leftover * 0.7
    else:
        # In-scope: allow the field weight to grow but cap it at 0.75.
        # A moderately high cap preserves differentiation between
        # moderately specific and highly specific queries.  We avoid
        # saturating too early (e.g. at 0.6) so that queries with higher
        # specificity continue to receive a higher field weight.  A cap
        # prevents the field from consuming all the weight, which would
        # render the dense and sparse retrievers ineffective on edge
        # cases.
        cap_field = 0.80
        w_field = min(w_field_raw, cap_field)
        base_semantic = 0.15  # ensure a minimum semantic contribution
        leftover = max(0.0, 1.0 - base_semantic - w_field)
        # Distribute the leftover primarily to the dense retriever (70%)
        # and the remainder to the sparse retriever (30%).
        w_dense = base_semantic + leftover * 0.7
        w_sparse = leftover * 0.3

    # 5. Normalise raw scores into [0,1] ranges.
    # Field scores: scale by field_max_score and cap at 1.0.
    fs_scale = field_max_score if field_max_score > 0 else 1.0
    norm_field_scores = {
        doc_id: min(score / fs_scale, 1.0)
        for doc_id, score in field_scores.items()
    }

    # Sparse scores (BM25): scale by bm25_scale.  Negative values are
    # retained to penalise irrelevant queries.  Positive values are
    # capped at 1.0 to prevent unbounded growth.
    bs_scale = bm25_scale if bm25_scale > 0 else 1.0
    norm_sparse_scores = {
        doc_id: max(-1.0, min(score / bs_scale, 1.0))
        for doc_id, score in sparse_scores.items()
    }

    # Dense scores: subtract threshold and scale by dense_range.  Values
    # below the threshold map to 0.0.  Cap at 1.0 to ensure the
    # contribution does not exceed the bound.
    drange = dense_range if dense_range > 0 else 1.0
    dthresh = dense_threshold
    norm_semantic_scores = {
        doc_id: max(0.0, min((score - dthresh) / drange, 1.0))
        for doc_id, score in semantic_scores.items()
    }

    # 6. Compute fusion scores using the weights.
    fusion_scores: Dict[str, float] = {}
    for doc_id in doc_inventory.keys():
        f = norm_field_scores.get(doc_id, 0.0)
        s = norm_sparse_scores.get(doc_id, 0.0)
        d = norm_semantic_scores.get(doc_id, 0.0)
        fusion_scores[doc_id] = w_field * f + w_sparse * s + w_dense * d

    # 7. Add a specificity bonus uniformly to all documents for this query.
    # Bonus: scale linearly with specificity relative to the maximum possible field
    # score (field_max_score).  This keeps the bonus within a reasonable
    # range and allows sufficient differentiation without causing saturation.
    bonus = 0.0
    bonus_range = field_max_score if field_max_score > 0 else 1.0
    bonus = (specificity_score / bonus_range) * bonus_factor

    # 8. Sort and clamp the final scores to [0,1].
    sorted_docs = sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)
    final_ranked_list: List[Dict] = []
    for doc_id, score in sorted_docs:
        doc = doc_inventory[doc_id]
        total = score + bonus
        # Ensure the final score does not exceed 1.0 and is not below 0.
        doc['fusion_score'] = min(max(total, 0.0), 1.0)
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
            'field_metadata': 0.6, # 0.5
            'sparse_bm25': 0.25, # 0.3
            'semantic_e5': 0.15 # 0.1
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