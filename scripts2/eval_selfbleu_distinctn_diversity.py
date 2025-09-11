# scripts/eval_combined_diversity.py

import json
import pandas as pd
from pathlib import Path
import sys
import nltk
import numpy as np

# --- Path and Environment Setup ---
# Ensures the script can find the custom evaluation utility modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Evaluator Imports ---
# Import the necessary classes from your project's evaluation utilities
from evaluation.utils.SelfBleu import SelfBleu
from evaluation.utils.distinct_n import DistinctnEvaluator

# Automatically download the 'punkt' tokenizer if it's not already installed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK 'punkt' data package...")
    nltk.download('punkt')


def run_combined_diversity_evaluation(input_filepath: Path):
    """
    Loads generated session plans, evaluates them using both Self-Bleu and
    a suite of Distinct-n metrics, and computes a final combined diversity score.
    """
    # --- 1. Load Data From Specified JSON File ---
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} items from '{input_filepath.name}'")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_filepath}'")
        sys.exit(1)

    # --- 2. Prepare DataFrame and Group by Corpus Type ---
    df = pd.DataFrame(data)
    # Extract grammar type (e.g., 'loose', 'balanced') from the 'case_id'
    df['grammar'] = df['case_id'].apply(lambda x: x.split('_')[0])
    grouped_by_grammar = df.groupby('grammar')

    print(f"\nFound {len(grouped_by_grammar)} grammar types to evaluate: {list(grouped_by_grammar.groups.keys())}")

    final_diversity_scores = {}

    # --- 3. Main Evaluation Loop (One Iteration Per Grammar Type) ---
    for grammar, group in grouped_by_grammar:
        print("\n" + "=" * 80)
        print(f"Processing grammar type: '{grammar.upper()}' ({len(group)} plans)")
        print("=" * 80)

        generated_texts = group['generated_plan'].tolist()

        # --- Metric A: Calculate Self-BLEU ---
        # Self-BLEU measures similarity; a lower score (closer to 0) indicates higher diversity.
        self_bleu_evaluator = SelfBleu(generated_texts=generated_texts, gram=4)
        self_bleu_score = self_bleu_evaluator.get_score(is_fast=True)
        print(f"  -> Intermediate Self-BLEU Score: {self_bleu_score:.4f}")

        # --- Metric B: Calculate Distinct-n Suite ---
        # Distinct-n measures lexical richness; a higher score (closer to 1) indicates higher diversity.
        distinct_evaluator = DistinctnEvaluator()
        distinct_scores = distinct_evaluator.evaluate(generated_texts)
        print("  -> Intermediate Distinct-n Scores:")
        for key, value in distinct_scores.items():
            print(f"     - {key}: {value:.4f}")

        # --- 4. Calculate the Final Combined Diversity Score using the CORRECTED formula ---
        p_d_1 = distinct_scores.get('prose_distinct_1', 0.0)
        p_d_2 = distinct_scores.get('prose_distinct_2', 0.0)
        patt_d = distinct_scores.get('pattern_diversity', 0.0)
        shot_d = distinct_scores.get('specific_shot_diversity_2', 0.0)

        # STEP 1: Average the distinct-n scores for a balanced contribution.
        average_distinct_score = (p_d_1 + p_d_2 + patt_d + shot_d) / 4.0
        print(f"  -> Intermediate Average Distinct Score: {average_distinct_score:.4f}")

        # STEP 2: Harmonize Self-BLEU by converting it to a diversity metric.
        self_bleu_diversity_score = 1 - self_bleu_score

        # STEP 3: Apply the corrected formula.
        diversity_score = np.sqrt(
            (0.5 * (self_bleu_diversity_score ** 2)) +
            (0.5 * (average_distinct_score ** 2))
        )

        final_diversity_scores[grammar] = diversity_score

    # --- 5. Output the Final Results ---
    print("\n" + "=" * 80)
    print("      FINAL COMBINED DIVERSITY SCORES")
    print("=" * 80)

    if not final_diversity_scores:
        print("No results to display.")
    else:
        for grammar, score in final_diversity_scores.items():
            print(f"  -> Overall Diversity Score for '{grammar.upper()}': {score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    # Define the absolute path to the JSON file to be evaluated
    INPUT_FILE = (
        Path("/experiments/selfbleu_distinctn/evaluation_sessions_set_k10_size499_20250907_192536.json"))

    # Run the main evaluation function
    run_combined_diversity_evaluation(INPUT_FILE)