# scripts/evaluate_diversity_with_distinctn.py
# Requirement: need the path of generated sessions (json)

import json
import pandas as pd
from pathlib import Path
import sys

# --- Path and Environment Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from evaluation.utils.distinct_n import DistinctnEvaluator


def run_distinct_n_evaluation(input_filepath: Path):
    """
    Loads generated plans and evaluates their diversity using Distinct-n metrics.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return

    # Use pandas to easily group the generated plans by their original query
    df = pd.DataFrame(data)
    grouped_by_query = df.groupby('query_text')

    evaluator = DistinctnEvaluator()
    all_results = []

    print("=" * 80)
    print(f"Analysing diversity for {len(grouped_by_query)} unique queries...")
    print("=" * 80)

    for query_text, group in grouped_by_query:
        print(f"\nProcessing query: \"{query_text[:60]}...\"")

        # Get the list of generated plans for this specific query
        generated_texts = group['generated_plan'].tolist()

        # Calculate the four diversity scores for this set of texts
        scores = evaluator.evaluate(generated_texts)

        print(f"  -> Prose Distinct-1: {scores['prose_distinct_1']:.4f}")
        print(f"  -> Prose Distinct-2: {scores['prose_distinct_2']:.4f}")
        print(f"  -> Pattern Distinct-2: {scores['pattern_distinct_2']:.4f}")
        print(f"  -> Pattern Distinct-3: {scores['pattern_distinct_3']:.4f}")

        scores['query_text'] = query_text
        all_results.append(scores)

    # --- FINAL SUMMARY ---
    summary_df = pd.DataFrame(all_results)
    avg_scores = summary_df.mean(numeric_only=True)

    print("\n" + "=" * 80)
    print("Final Diversity Evaluation Summary (Averaged Across All Queries)")
    print("=" * 80)
    print(avg_scores.round(4))

    # Optionally, save the detailed results to a CSV
    # summary_df.to_csv("distinct_n_results.csv", index=False)


if __name__ == "__main__":
    # Define the file containing the generated plans you want to analyse
    # This should be the output from your 'generate_qualitative_set.py' script
    INPUT_FILE = PROJECT_ROOT / "human_evaluation_set_k3_size100.json"
    run_distinct_n_evaluation(INPUT_FILE)