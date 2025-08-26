# evaluation/corpus_analysis/statistics/measure_diversity.py

# - Lexical Diversity (vocabulary size)
# - Structural Diversity (Combinatorial Variety)
# - Content Similarity (uniqueness of sessions)

import argparse
import math
from pathlib import Path
from collections import Counter
from typing import List
from evaluation.corpus_analysis.utils import load_corpus, count_total_variants

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
def calculate_shannon_entropy(all_variants_used: List[str]) -> float:
    """Calculates the Shannon Entropy of the variant distribution."""
    if not all_variants_used:
        return 0.0

    variant_counts = Counter(all_variants_used)
    total_occurrences = len(all_variants_used)
    probabilities = [count / total_occurrences for count in variant_counts.values()]

    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

# --- Structural Diversity (Combinatorial Variety) ---
# --- Content Similarity (uniqueness of sessions) ---


# --- Main Analysis Orchestrator ---

def analyse_lexical_diversity(corpus_path: Path, grammar_profile: str):
    """
    Orchestrates the lexical diversity analysis for a given corpus.
    """
    corpus = load_corpus(corpus_path)
    if not corpus:
        return

    all_variants_used = [
        exercise.get("variant_id")
        for session in corpus
        for exercise in session.get("meta", {}).get("exercises_used", [])
        if exercise and exercise.get("variant_id")
    ]

    if not all_variants_used:
        print("No variants found in corpus.")
        return

    # --- Calculate Metrics using Helper Functions ---
    unique_variants_used = len(set(all_variants_used))
    total_variants_defined = count_total_variants(grammar_profile)

    library_coverage = calculate_library_coverage(unique_variants_used, total_variants_defined)
    entropy = calculate_shannon_entropy(all_variants_used)

    # --- Print Report ---
    print("\n--- Advanced Lexical Diversity Report ---")
    print(f"Corpus: {corpus_path.parent.name}")
    print(f"\n[Library Coverage]")
    print(f"  - Unique Variants Used: {unique_variants_used}")
    print(f"  - Total Variants Defined in Grammar: {total_variants_defined}")
    print(f"  - Coverage: {library_coverage:.2f}%")

    print(f"\n[Variant Distribution Entropy]")
    print(f"  - Shannon Entropy: {entropy:.4f} bits")
    print("  (Higher entropy means more variety and less predictability)")
    print("-----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the lexical diversity of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("grammar_profile", type=str,
                        help="Name of the grammar profile (e.g., 'high_constraint_grammar').")
    args = parser.parse_args()

    analyse_lexical_diversity(args.corpus_path, args.grammar_profile)