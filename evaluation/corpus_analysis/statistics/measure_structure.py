# evaluation/corpus_analysis/_3_statistics/measure_structure.py


"""
DESCRIPTION

Statistical Distributions:
  The first part of the script performs simple counts of the high-level archetype and structure_id components,
  giving you a clear picture of the compositional makeup of each corpus.

N-gram / Transition Analysis:
  The second, more advanced part models the flow of the session. It measures how predictable the next exercise
  family is, given the current one. Low transition entropy is a strong indicator of a highly structured, predictable
  grammar, while high entropy indicates a random, less predictable flow.
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any

# Import the shared utility and metric functions
from evaluation.corpus_analysis.utils import load_corpus, count_total_variants
from evaluation.corpus_analysis.statistics.common_metrics import calculate_shannon_entropy


# --- Metric Helper Functions ---

def calculate_distributions(corpus: List[Dict]) -> tuple[Counter, Counter]:
    """Counts the occurrences of archetypes and structure IDs in a corpus."""
    archetype_counts = Counter()
    structure_counts = Counter()

    for session in corpus:
        meta = session.get("meta", {})
        if archetype := meta.get("archetype"):
            archetype_counts[archetype] += 1
        if structure_id := meta.get("structure_id"):
            structure_counts[structure_id] += 1

    return archetype_counts, structure_counts


def analyse_family_transitions(corpus: List[Dict]) -> Dict[str, Any]:
    """
    Performs an N-gram (bigram) analysis of exercise family transitions.
    Calculates transition probabilities and the entropy of the distribution.
    """
    transition_counts = defaultdict(Counter)

    for session in corpus:
        # Extract the sequence of family IDs for this session
        family_sequence = [
            ex.get("family_id")
            for ex in session.get("meta", {}).get("exercises_used", [])
            if ex and ex.get("family_id")
        ]

        # Create and count bigrams (adjacent pairs)
        for i in range(len(family_sequence) - 1):
            current_family = family_sequence[i]
            next_family = family_sequence[i + 1]
            transition_counts[current_family][next_family] += 1

    if not transition_counts:
        return {"top_transitions": [], "transition_entropy": 0.0}

    # Calculate transition probabilities
    transition_probabilities = defaultdict(dict)
    all_probabilities = []
    for current_family, next_family_counts in transition_counts.items():
        total_transitions_from_current = sum(next_family_counts.values())
        for next_family, count in next_family_counts.items():
            prob = count / total_transitions_from_current
            transition_probabilities[current_family][next_family] = prob
            all_probabilities.append(prob)

    # Calculate the overall entropy of the transition distribution
    entropy = calculate_shannon_entropy(all_probabilities)

    # Find the most common transitions for the report
    all_transitions_flat = [
        (current, next, data)
        for current, next_data in transition_probabilities.items()
        for next, data in next_data.items()
    ]
    top_transitions = sorted(all_transitions_flat, key=lambda item: item[2], reverse=True)[:5]

    return {"top_transitions": top_transitions, "transition_entropy": entropy}


# --- Main Analysis Orchestrator ---

def analyse_structure_metrics(corpus_path: Path) -> dict:
    """Orchestrates the structural analysis for a given corpus."""
    corpus = load_corpus(corpus_path)
    if not corpus:
        return {}

    archetype_counts, structure_counts = calculate_distributions(corpus)
    transition_analysis = analyse_family_transitions(corpus)


    # --- Print Report ---
    # print("\n--- Structural Analysis Report ---")
    # print(f"Corpus: {corpus_path.parent.name}")
    #
    # print("\n[Archetype Distribution]")
    # for archetype, count in archetype_counts.most_common():
    #     print(f"  - {archetype}: {count} sessions ({count / len(corpus) * 100:.1f}%)")
    #
    # print("\n[Session Structure Distribution]")
    # for structure, count in structure_counts.most_common():
    #     print(f"  - {structure}: {count} sessions ({count / len(corpus) * 100:.1f}%)")
    #
    # print("\n[Exercise Family Transition Analysis]")
    # print(f"  - Transition Entropy: {transition_analysis['transition_entropy']:.4f} bits")
    # print("    (Lower entropy means more predictable session flow)")
    # print("\n  - Top 5 Most Probable Transitions:")
    # for current, next_fam, prob in transition_analysis['top_transitions']:
    #     print(f"    - P({next_fam.split('.')[-1]} | {current.split('.')[-1]}): {prob:.2f}")
    #
    # print("----------------------------------")

    # Return results as a dictionary
    return {
        "archetype_distribution": dict(archetype_counts),
        "structure_distribution": dict(structure_counts),
        "transition_entropy_bits": transition_analysis['transition_entropy']
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the structural properties of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    args = parser.parse_args()

    analyse_structure_metrics(args.corpus_path)