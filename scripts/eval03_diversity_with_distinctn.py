# scripts/eval03_diversity_with_distinctn.py

import json
import pandas as pd
from pathlib import Path
import sys
import nltk

# --- Path and Environment Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Automatically download 'punkt' if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK 'punkt' data package...")
    nltk.download('punkt')

# FIX 3: Ensure the correct, updated evaluator is imported
from evaluation.utils.distinct_n import DistinctnEvaluator


def run_distinct_n_evaluation(input_filepath: Path):
    """
    Loads generated plans and evaluates their diversity using a suite of Distinct-n metrics.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return

    df = pd.DataFrame(data)
    grouped_by_query = df.groupby('query_text')

    evaluator = DistinctnEvaluator()
    all_results = []

    print("=" * 80)
    print(f"Analysing diversity for {len(grouped_by_query)} unique queries...")
    print("=" * 80)

    for query_text, group in grouped_by_query:
        print(f"\nProcessing query: \"{query_text[:60]}...\"")
        generated_texts = group['generated_plan'].tolist()
        scores = evaluator.evaluate(generated_texts)

        # FIX 3: Use the correct keys returned by the updated evaluator
        print(f"  -> Prose Distinct-1: {scores['prose_distinct_1']:.4f}")
        print(f"  -> Prose Distinct-2: {scores['prose_distinct_2']:.4f}")
        print(f"  -> Pattern Diversity: {scores['pattern_diversity']:.4f}")
        print(f"  -> Specific Shot Diversity-2: {scores['specific_shot_diversity_2']:.4f}")

        scores['query_text'] = query_text
        all_results.append(scores)

    # --- FINAL SUMMARY ---
    summary_df = pd.DataFrame(all_results)
    avg_scores = summary_df.mean(numeric_only=True)

    print("\n" + "=" * 80)
    print("Final Diversity Evaluation Summary (Averaged Across All Queries)")
    print("=" * 80)

    # FIX 2: Removed the invalid .set_index('grammar_type') call
    print(avg_scores.round(4))

    # Optionally, save the detailed results to a CSV
    summary_df.to_csv("distinct_n_results.csv", index=False)
    print("\n✅ Detailed diversity results saved to 'distinct_n_results.csv'")


if __name__ == "__main__":
    # FIX 1: Ensure this filename matches the JSON file produced by your generation script
    INPUT_FILE = PROJECT_ROOT / "generated_sessions_12queries_set_k3_size100.json"
    run_distinct_n_evaluation(INPUT_FILE)