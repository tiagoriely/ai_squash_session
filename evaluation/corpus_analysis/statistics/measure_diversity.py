# evaluation/corpus_analysis/statistics/measure_diversity.py

# - Lexical Diversity (vocabulary size)
# - Structural Diversity (Combinatorial Variety)
# - Content Similarity (uniqueness of sessions)

import argparse
import math
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List
from evaluation.corpus_analysis.utils import load_corpus, count_total_variants

from .common_metrics import calculate_shannon_entropy


# --- Lexical Diversity (vocabulary size) ---
# 1. Coverage
# 2. Entropy

# Coverage
def calculate_library_coverage(unique_variants_used: int, total_variants_defined: int) -> float:
    """Calculates the percentage of the grammar's variant library that was used."""
    if total_variants_defined == 0:
        return 0.0
    return (unique_variants_used / total_variants_defined) * 100

# Entropy
def calculate_variant_distribution_entropy(all_variants_used: List[str]) -> float:
    """Calculates the Shannon Entropy of the variant distribution."""
    if not all_variants_used:
        return 0.0

    variant_counts = Counter(all_variants_used)
    total_occurrences = len(all_variants_used)
    # First, calculate the probabilities of each variant
    probabilities = [count / total_occurrences for count in variant_counts.values()]

    # Then, pass the probabilities to the common entropy function
    return calculate_shannon_entropy(probabilities)

# --- Structural Diversity (Combinatorial Variety) ---
def calculate_intra_session_family_diversity(corpus: List[dict]) -> dict:
    """Calculates the average number of unique exercise families used per session."""
    family_counts_per_session = []
    for session in corpus:
        exercises = session.get("meta", {}).get("exercises_used", [])
        if exercises:
            session_families = {ex.get("family_id") for ex in exercises if ex.get("family_id")}
            family_counts_per_session.append(len(session_families))

    if not family_counts_per_session:
        return {'mean': 0, 'std': 0}

    return {
        'mean': np.mean(family_counts_per_session),
        'std': np.std(family_counts_per_session)
    }
# --- Content Similarity (uniqueness of sessions) ---


# --- Main Analysis Orchestrator ---

def analyse_diversity_metrics(corpus_path: Path, grammar_profile: str) -> dict:
    """
    Orchestrates the statistical diversity analysis for a given corpus.
    """
    corpus = load_corpus(corpus_path)
    if not corpus:
        return {}

    all_variants_used = [
        exercise.get("variant_id")
        for session in corpus
        for exercise in session.get("meta", {}).get("exercises_used", [])
        if exercise and exercise.get("variant_id")
    ]

    # --- Calculate All Metrics ---
    # Lexical
    unique_variants_used = len(set(all_variants_used))
    total_variants_defined = count_total_variants(grammar_profile)
    library_coverage = calculate_library_coverage(unique_variants_used, total_variants_defined)
    entropy = calculate_variant_distribution_entropy(all_variants_used)

    # Structural
    family_diversity = calculate_intra_session_family_diversity(corpus)

    # # --- Print Combined Report ---
    # print("\n--- Statistical Diversity Report ---")
    # print(f"Corpus: {corpus_path.parent.name}")
    #
    # print(f"\n[Lexical Diversity]")
    # print(f"  - Coverage: {library_coverage:.2f}% ({unique_variants_used} of {total_variants_defined} variants used)")
    # print(f"  - Shannon Entropy: {entropy:.4f} bits")
    #
    # print(f"\n[Structural Diversity]")
    # print(f"  - Avg. Unique Families per Session: {family_diversity['mean']:.2f} (std: {family_diversity['std']:.2f})")
    #
    # print("------------------------------------")

    # Return results as a dictionary
    return {
        "library_coverage_percent": library_coverage,
        "unique_variants_used": unique_variants_used,
        "total_variants_defined": total_variants_defined,
        "variant_entropy_bits": entropy,
        "avg_families_per_session": family_diversity['mean'],
        "std_families_per_session": family_diversity['std']
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the statistical diversity of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("grammar_profile", type=str,
                        help="Name of the grammar profile (e.g., 'high_constraint_grammar').")

    args = parser.parse_args()

    analyse_diversity_metrics(args.corpus_path, args.grammar_profile)
