# scripts/eval_selfbleu_distinctn_diversity.py

import json
import pandas as pd
from pathlib import Path
import sys
import nltk
import numpy as np
import re # Imported for regular expression matching

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
    a suite of Distinct-n metrics, computes a final combined diversity score,
    and returns the results for aggregation.
    """
    # --- 1. Load Data and Extract Metadata from Filename ---
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} items from '{input_filepath.name}'")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_filepath}'")
        return []

    # Extract 'size' from the filename using a regular expression
    filename = input_filepath.name
    size_match = re.search(r'size(\d+)_', filename)
    # Use a default value (e.g., 0) if the pattern isn't found
    extracted_size = int(size_match.group(1)) if size_match else 0
    if extracted_size:
        print(f"  -> Extracted size: {extracted_size}")


    # --- 2. Prepare DataFrame and Group by Corpus Type ---
    df = pd.DataFrame(data)
    # Extract grammar type (e.g., 'loose', 'balanced') from the 'case_id'
    df['grammar'] = df['case_id'].apply(lambda x: x.split('_')[0])
    grouped_by_grammar = df.groupby('grammar')

    print(f"\nFound {len(grouped_by_grammar)} grammar types to evaluate: {list(grouped_by_grammar.groups.keys())}")

    final_diversity_scores = {}
    # List to store result dictionaries for CSV export
    results_for_csv = []

    # --- 3. Main Evaluation Loop (One Iteration Per Grammar Type) ---
    for grammar, group in grouped_by_grammar:
        print("\n" + "=" * 80)
        print(f"Processing grammar type: '{grammar.upper()}' ({len(group)} plans)")
        print("=" * 80)

        generated_texts = group['generated_plan'].tolist()

        # --- Metric A: Calculate Self-BLEU ---
        self_bleu_evaluator = SelfBleu(generated_texts=generated_texts, gram=4)
        self_bleu_score = self_bleu_evaluator.get_score(is_fast=True)
        print(f"  -> Intermediate Self-BLEU Score: {self_bleu_score:.4f}")

        # --- Metric B: Calculate Distinct-n Suite ---
        distinct_evaluator = DistinctnEvaluator()
        distinct_scores = distinct_evaluator.evaluate(generated_texts)
        print("  -> Intermediate Distinct-n Scores:")
        for key, value in distinct_scores.items():
            print(f"     - {key}: {value:.4f}")

        # --- 4. Calculate the Final Combined Diversity Score ---
        p_d_1 = distinct_scores.get('prose_distinct_1', 0.0)
        p_d_2 = distinct_scores.get('prose_distinct_2', 0.0)
        patt_d = distinct_scores.get('pattern_diversity', 0.0)
        shot_d = distinct_scores.get('specific_shot_diversity_2', 0.0)

        average_distinct_score = (p_d_1 + p_d_2 + patt_d + shot_d) / 4.0
        print(f"  -> Intermediate Average Distinct Score: {average_distinct_score:.4f}")

        self_bleu_diversity_score = 1 - self_bleu_score
        diversity_score = np.sqrt(
            (0.5 * (self_bleu_diversity_score ** 2)) +
            (0.5 * (average_distinct_score ** 2))
        )

        final_diversity_scores[grammar] = diversity_score

        # --- Store results for this grammar type for later CSV export ---
        results_for_csv.append({
            'source_file': filename,
            'grammar_type': grammar,
            'size': extracted_size, # Add the extracted size to the results
            'avg_distinct_n': average_distinct_score,
            'self_bleu': self_bleu_score,
            'combined_diversity_score': diversity_score
        })

    # --- 5. Output the Final Results to Console ---
    print("\n" + "=" * 80)
    print("      FINAL COMBINED DIVERSITY SCORES (CONSOLE)")
    print("=" * 80)

    if not final_diversity_scores:
        print("No results to display.")
    else:
        for grammar, score in final_diversity_scores.items():
            print(f"  -> Overall Diversity Score for '{grammar.upper()}': {score:.4f}")
    print("=" * 80)

    return results_for_csv


if __name__ == "__main__":
    TARGET_DIR = PROJECT_ROOT / "experiments" / "dynamic_fusion_retrieval"

    if not TARGET_DIR.is_dir():
        print(f"❌ Error: Target directory not found at '{TARGET_DIR}'")
        sys.exit(1)

    # Find all JSON files in the directory that match the expected pattern
    json_files_to_process = list(TARGET_DIR.glob('evaluation_sessions_set_*.json'))

    if not json_files_to_process:
        print(f"🤷 No matching JSON files found in '{TARGET_DIR}'. Nothing to process.")
        sys.exit(0)

    print(f"Found {len(json_files_to_process)} JSON file(s) to process in the target directory.")

    all_results_aggregator = []

    for filepath in sorted(json_files_to_process): # Sorting helps keep output consistent
        print("\n\n" + "#" * 80)
        print(f"   PROCESSING FILE: {filepath.name}")
        print("#" * 80)
        file_results = run_combined_diversity_evaluation(filepath)
        if file_results:
            all_results_aggregator.extend(file_results)

    # --- Write aggregated results to a single CSV file ---
    if not all_results_aggregator:
        print("\n\n🤷 No results were generated to save.")
    else:
        results_df = pd.DataFrame(all_results_aggregator)
        output_csv_path = TARGET_DIR / "diversity_evaluation_summary.csv"

        column_order = [
            'source_file',
            'grammar_type',
            'size',
            'avg_distinct_n',
            'self_bleu',
            'combined_diversity_score'
        ]
        results_df = results_df[column_order]
        # Sort the final CSV for better readability
        results_df = results_df.sort_values(by=['size', 'grammar_type']).reset_index(drop=True)

        try:
            results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
            print("\n\n" + "*" * 80)
            print(f"📊 Summary of all results saved to: {output_csv_path}")
            print("*" * 80)
        except Exception as e:
            print(f"\n\n❌ Error saving results to CSV: {e}")

    print("\n\n✅ All files processed.")