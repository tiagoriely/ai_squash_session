# evaluation/corpus_analysis/statistics/measure_diversity.py

"""
Diversity metrics over generated squash training corpora.

This module quantifies diversity along three axes:

A) Corpus-level per-occurrence diversity for variants & families:
   H, H' (=H/log2 K), ^1D (=2^H), Simpson (1-Σ p_i^2).

B) Intra-session richness & evenness:
   R_fam, R_var, TTR per session; mean/std across sessions.

C) Session uniqueness:
   Mean/std pairwise Jaccard distance over session variant sets
   at three layers: raw, side-normalised, and base-normalised (side+mode removed).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evaluation.corpus_analysis.utils import load_corpus, count_total_variants
from evaluation.corpus_analysis.statistics.common_metrics import calculate_shannon_entropy

# Remove side tokens from IDs to avoid artificial inflation from forehand/backhand mirroring.
SIDE_TOKENS = re.compile(r'(?i)\b(?:forehand|backhand)\b')

# collapse both side and mode markers to a "base" variant id
MODE_TOKENS = re.compile(r'(?i)\b(?:drill|cg|conditioned|game|conditioned_game)\b')


def _normalise_variant_side(variant_id: str) -> str:
    """
    Remove explicit side tokens ('forehand'/'backhand') from a variant id.

    Examples:
      'deep_only_backhand_conditioned_game' -> 'deep_only_conditioned_game'
      'volley_drop_deep_only_forehand_drill'-> 'volley_drop_deep_only_drill'
    """
    if not variant_id:
        return variant_id
    tokens = re.split(r'[^a-zA-Z0-9]+', variant_id)
    tokens = [t for t in tokens if t and not SIDE_TOKENS.fullmatch(t)]
    return "_".join(tokens)

def _canonicalise_variant_id(variant_id: str) -> str:
    """
    Return a base variant id with side and mode tokens removed.
    Examples:
      'deep_onlyD_forehand_drill' -> 'deep_onlyd'
      'counter_drop_backhand_conditioned_game' -> 'counter_drop'
    """
    if not variant_id:
        return variant_id
    tokens = re.split(r'[^a-zA-Z0-9]+', variant_id)
    out = []
    for t in tokens:
        if not t:
            continue
        low = t.lower()
        if SIDE_TOKENS.fullmatch(low):
            continue
        if MODE_TOKENS.fullmatch(low):
            continue
        out.append(low)
    return "_".join(out)

def _iter_activity_sequences(corpus: Iterable[dict]) -> Iterable[Tuple[str, str]]:
    for session in corpus:
        for ex in (session.get("meta", {}) or {}).get("exercise_sequences", []) or []:
            fam = ex.get("exercise_family_id")
            var = ex.get("exercise_variant_id")
            if not fam or fam == "squash.family.warmup":
                continue
            if not var:
                continue
            yield fam, var

@dataclass
class DistributionStats:
    counts: Dict[str, int]
    total: int
    distinct: int
    entropy_bits: float
    entropy_norm: float
    hill_number: float
    simpson: float
    top_k: List[Tuple[str, int]]

def _distribution_stats(items: Iterable[str], top_k: int = 10) -> DistributionStats:
    counts = Counter(items)
    N = sum(counts.values())
    K = len(counts)
    if N == 0 or K == 0:
        return DistributionStats(counts={}, total=0, distinct=0,
                                 entropy_bits=0.0, entropy_norm=0.0,
                                 hill_number=0.0, simpson=0.0, top_k=[])
    probs = [c / N for c in counts.values()]
    H = calculate_shannon_entropy(probs)
    Hmax = math.log2(K) if K > 1 else 0.0
    Hprime = (H / Hmax) if Hmax > 0 else 0.0
    simpson = 1.0 - sum(p * p for p in probs)
    return DistributionStats(
        counts=dict(counts),
        total=N,
        distinct=K,
        entropy_bits=H,
        entropy_norm=Hprime,
        hill_number=(2.0 ** H),
        simpson=simpson,
        top_k=sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k],
    )

def _intra_session_richness(corpus: Iterable[dict]) -> Dict[str, float]:
    fam_rich, var_rich, ttr = [], [], []
    for session in corpus:
        fams, vars_base = [], []
        for fam, var in _iter_activity_sequences([session]):
            fams.append(fam)
            vars_base.append(_canonicalise_variant_id(var))
        fam_set, var_set = set(fams), set(vars_base)
        fam_rich.append(len(fam_set))
        var_rich.append(len(var_set))
        denom = len(vars_base) if vars_base else 1
        ttr.append(len(var_set) / denom)
    # means and population stds
    from statistics import mean, pstdev
    def _msd(arr):
        return (0.0, 0.0) if not arr else (float(arr[0]), 0.0) if len(arr)==1 else (mean(arr), pstdev(arr))
    m_fam,s_fam=_msd(fam_rich); m_var,s_var=_msd(var_rich); m_ttr,s_ttr=_msd(ttr)
    return {
        "mean_family_richness": m_fam, "std_family_richness": s_fam,
        "mean_variant_richness": m_var, "std_variant_richness": s_var,
        "mean_ttr": m_ttr, "std_ttr": s_ttr, "sessions": len(fam_rich),
    }

def _per_session_sets(corpus: Iterable[dict], normalise_side: bool) -> List[set]:
    sets = []
    for session in corpus:
        s = set()
        for _, var in _iter_activity_sequences([session]):
            s.add(_normalise_variant_side(var) if normalise_side else var)
        sets.append(s)
    return sets

def _pairwise_jaccard_stats(sets: Sequence[set], cap_pairs: int = 50000, rng_seed: int = 13) -> Dict[str, float]:
    n = len(sets)
    if n < 2:
        return {"mean": 0.0, "std": 0.0, "pairs": 0}
    pairs = [(i, j) for i in range(n-1) for j in range(i+1, n)]
    total_pairs = len(pairs)
    if total_pairs > cap_pairs:
        import random
        random.Random(rng_seed).shuffle(pairs)
        pairs = pairs[:cap_pairs]
    dists = []
    for i, j in pairs:
        A, B = sets[i], sets[j]
        if not A and not B:
            dists.append(0.0); continue
        J = (len(A & B) / len(A | B)) if (A or B) else 0.0
        dists.append(1.0 - J)
    from statistics import mean, pstdev
    return {"mean": mean(dists) if dists else 0.0,
            "std": pstdev(dists) if len(dists)>1 else 0.0,
            "pairs": len(dists), "total_pairs": total_pairs}


def _semantic_diversity(corpus: Iterable[dict], model_name: str = 'all-MiniLM-L6-v2') -> Dict[str, float]:
    """
    Calculates the semantic diversity of a corpus based on exercise variant descriptions.

    This is done by embedding all unique variant descriptions and then computing the
    average pairwise cosine similarity. A lower similarity score indicates higher
    semantic diversity.

    Returns:
        A dictionary containing the mean similarity and standard deviation.
    """
    # Lazily load the model to avoid overhead when not used
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Could not load SentenceTransformer model '{model_name}'. Is it installed? Error: {e}")
        return {"mean_similarity": 1.0, "std_similarity": 0.0, "unique_variants": 0}

    # Extract unique variant descriptions to avoid redundant embedding
    unique_variants = set(var for _, var in _iter_activity_sequences(corpus))

    if len(unique_variants) < 2:
        return {"mean_similarity": 0.0, "std_similarity": 0.0, "unique_variants": len(unique_variants)}

    # Generate embeddings
    embeddings = model.encode(list(unique_variants), show_progress_bar=False)

    # Calculate pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Extract the upper triangle of the similarity matrix (excluding the diagonal)
    # to get unique pairwise scores
    indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_scores = sim_matrix[indices]

    if pairwise_scores.size == 0:
        return {"mean_similarity": 0.0, "std_similarity": 0.0, "unique_variants": len(unique_variants)}

    return {
        "mean_similarity": float(np.mean(pairwise_scores)),
        "std_similarity": float(np.std(pairwise_scores)),
        "unique_variants": len(unique_variants),
    }


# --- Main Analysis Orchestrator ---

def analyse_diversity_metrics(corpus: List[dict], grammar_profile: str | None = None) -> Dict[str, Dict]:
    # build per-occurrence lists
    variant_occ, family_occ = [], []
    for fam, var in _iter_activity_sequences(corpus):
        family_occ.append(fam); variant_occ.append(var)

    variant_occ_norm = [_normalise_variant_side(v) for v in variant_occ]
    variant_occ_base = [_canonicalise_variant_id(v) for v in variant_occ]

    var_raw = _distribution_stats(variant_occ)
    var_norm = _distribution_stats(variant_occ_norm)  # side-normalised
    var_base = _distribution_stats(variant_occ_base)  # side+mode-normalised

    fam_stats= _distribution_stats(family_occ)

    distinct_raw = len(set(variant_occ))
    distinct_norm = len(set(variant_occ_norm))
    distinct_base = len(set(variant_occ_base))
    coverage = {
        "distinct_variants_raw": distinct_raw,
        "distinct_variants_side_norm": distinct_norm,
        "distinct_variants_base_norm": distinct_base,
    }

    intra = _intra_session_richness(corpus)


    try:
        total_library = count_total_variants(grammar_profile)
        if isinstance(total_library, dict):
            # Expect keys like {'raw': K_raw, 'side': K_side, 'base': K_base}
            K_raw = total_library.get("raw") or 0
            K_side = total_library.get("side") or 0
            K_base = total_library.get("base") or 0
            if K_raw:  coverage["coverage_percent_raw"] = 100.0 * distinct_raw / K_raw
            if K_side: coverage["coverage_percent_side_norm"] = 100.0 * distinct_norm / K_side
            if K_base: coverage["coverage_percent_base_norm"] = 100.0 * distinct_base / K_base
    except Exception:
        pass

    # Per-session sets
    def _per_session_sets_base(corpus: Iterable[dict]) -> List[set]:
        sets = []
        for session in corpus:
            s = set()
            for _, var in _iter_activity_sequences([session]):
                s.add(_canonicalise_variant_id(var))
            sets.append(s)
        return sets

    sets_raw = _per_session_sets(corpus, normalise_side=False)
    sets_norm = _per_session_sets(corpus, normalise_side=True)
    sets_base = _per_session_sets_base(corpus)

    jacc_raw = _pairwise_jaccard_stats(sets_raw)
    jacc_norm = _pairwise_jaccard_stats(sets_norm)
    jacc_base = _pairwise_jaccard_stats(sets_base)

    semantic_div = _semantic_diversity(corpus)


    return {
        "variant_raw": {
            "total": var_raw.total, "distinct": var_raw.distinct,
            "entropy_bits": var_raw.entropy_bits, "entropy_norm": var_raw.entropy_norm,
            "hill_number": var_raw.hill_number, "simpson": var_raw.simpson,
            "top_variants": var_raw.top_k,
        },
        "variant_side_norm": {
            "total": var_norm.total, "distinct": var_norm.distinct,
            "entropy_bits": var_norm.entropy_bits, "entropy_norm": var_norm.entropy_norm,
            "hill_number": var_norm.hill_number, "simpson": var_norm.simpson,
            "top_variants": var_norm.top_k,
        },
        "variant_base_norm": {
            "total": var_base.total, "distinct": var_base.distinct,
            "entropy_bits": var_base.entropy_bits, "entropy_norm": var_base.entropy_norm,
            "hill_number": var_base.hill_number, "simpson": var_base.simpson,
            "top_variants": var_base.top_k,
        },
        "family": {
            "total": fam_stats.total, "distinct": fam_stats.distinct,
            "entropy_bits": fam_stats.entropy_bits, "entropy_norm": fam_stats.entropy_norm,
            "hill_number": fam_stats.hill_number, "simpson": fam_stats.simpson,
            "top_families": fam_stats.top_k,
        },
        "coverage": coverage,
        "intra_session": intra,
        "session_jaccard": {"raw": jacc_raw, "side_norm": jacc_norm, "base_norm": jacc_base},
        "semantic_diversity": semantic_div,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the statistical diversity of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("--grammar-profile", type=str, default=None,
                        help="Optional grammar profile name for coverage lookups (e.g., 'high_constraint').")
    parser.add_argument("--json-indent", type=int, default=2,
                        help="Pretty-print JSON with this indent.")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    results = analyse_diversity_metrics(corpus, grammar_profile=args.grammar_profile)

    print(json.dumps(results, indent=args.json_indent, ensure_ascii=False))