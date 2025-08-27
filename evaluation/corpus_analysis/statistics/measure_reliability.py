# evaluation/corpus_analysis/statistics/measure_reliability.py
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import List, Dict

from evaluation.corpus_analysis.utils import load_corpus, load_exercise_library


# --- Metric Helper Functions ---

def check_archetype_adherence(corpus: List[Dict]) -> Dict[str, float]:
    """
    Performs a post-mortem validation to check if sessions adhere to archetype rules.
    """
    results = defaultdict(lambda: {"success": 0, "total": 0})

    for session in corpus:
        meta = session.get("meta", {})
        archetype = meta.get("archetype", "")
        exercises = meta.get("exercises_used", [])
        if not exercises:
            continue

        # Check for "Drill Only" adherence
        if "Drill Only" in archetype:
            results["Drill Only Adherence"]["total"] += 1

            is_adherent = all("drill" in ex.get("types", []) for ex in exercises)
            if is_adherent:
                results["Drill Only Adherence"]["success"] += 1


        # Check for "Conditioned Games Only" adherence
        elif "Conditioned Games Only" in archetype:
            results["Conditioned Games Only Adherence"]["total"] += 1
            is_adherent = all("conditioned_game" in ex.get("types", []) for ex in exercises)
            if is_adherent:
                results["Conditioned Games Only Adherence"]["success"] += 1

        # Check for "Progressive Family" adherence
        elif "Progressive Family" in archetype:
            results["Single Family Adherence"]["total"] += 1
            family_ids = {ex.get("family_id") for ex in exercises if ex.get("family_id")}
            if len(family_ids) == 1:
                results["Single Family Adherence"]["success"] += 1

        # Check for "Progressive Single ShotSide" adherence
        elif "Single ShotSide" in archetype:
            results["Single Side Adherence"]["total"] += 1
            sides = {tuple(ex.get("shotSide", [])) for ex in exercises}
            if len(sides) == 1:
                results["Single Side Adherence"]["success"] += 1

    # Calculate final adherence rates
    adherence_rates = {
        name: (data["success"] / data["total"]) * 100 if data["total"] > 0 else 100
        for name, data in results.items()
    }
    return adherence_rates


def calculate_progression_scores(corpus: List[Dict], exercise_library: Dict) -> List[float]:
    """Calculates the difficulty progression score for each session in the corpus."""
    session_scores = []
    for session in corpus:
        variant_ids = [
            ex.get("variant_id")
            for ex in session.get("meta", {}).get("exercises_used", [])
            if ex and ex.get("variant_id")
        ]

        difficulty_scores = [
            exercise_library.get(vid, {}).get("difficulty_score", 0)
            for vid in variant_ids
        ]

        if len(difficulty_scores) < 2:
            continue

        progressions = 0
        total_steps = len(difficulty_scores) - 1
        for i in range(total_steps):
            if difficulty_scores[i + 1] >= difficulty_scores[i]:
                progressions += 1

        session_scores.append(progressions / total_steps)

    return session_scores


# --- Main Analysis Orchestrator ---

def analyse_reliability_metrics(corpus_path: Path, grammar_profile: str) -> dict:
    """Orchestrates the reliability analysis for a given corpus."""
    corpus = load_corpus(corpus_path)
    exercise_library = load_exercise_library(grammar_profile)
    if not corpus:
        return {}

    adherence_rates = check_archetype_adherence(corpus)
    progression_scores = calculate_progression_scores(corpus, exercise_library)

    # print("\n--- Reliability Analysis Report ---")
    # print(f"Corpus: {corpus_path.parent.name}")
    #
    # print("\n[Archetype Constraint Adherence]")
    # if not adherence_rates:
    #     print("  - No applicable archetype constraints found to check.")
    # for name, rate in adherence_rates.items():
    #     print(f"  - {name}: {rate:.2f}%")
    #
    # print("\n[Difficulty Progression Score]")
    # if not progression_scores:
    #     print("  - No sessions with sufficient length to calculate progression.")
    # else:
    #     print(f"  - Average Progression Score: {np.mean(progression_scores):.2f}")
    #     print("    (1.0 = perfectly non-decreasing difficulty)")

    # print("-----------------------------------")

    return {
        "adherence_rates_percent": adherence_rates,
        "avg_progression_score": np.mean(progression_scores) if progression_scores else 0.0
    }




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the reliability of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("grammar_profile", type=str,
                        help="Name of the grammar profile (e.g., 'high_constraint_grammar').")
    args = parser.parse_args()

    analyse_reliability_metrics(args.corpus_path, args.grammar_profile)