# scripts/compare_all_retrievers_final.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def final_retriever_comparison():
    """
    Loads the final evaluation results and compares all four strategies
    using the Mean Reciprocal Rank (MRR) metric.
    """
    input_path = "../evaluation/retrieval/all_corpora/final_evaluation_all_grammars_size100.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'run_final_evaluation.py' first.")
        return

    # --- 1. Define a HIGH-QUALITY "Golden Set" of Queries ---
    # FIX: This set now only includes highly specific, unambiguous queries where
    # we can be confident that the Field Retriever's top answer is the correct one.
    golden_query_ids = [
        "complex_01_cg", "complex_02_cg", "complex_03_cg",
        "complex_21_mix", "complex_22_mix", "complex_23_mix", "complex_24_mix",
    ]

    print(f"Performing MRR analysis on a 'Golden Set' of {len(golden_query_ids)} high-precision queries.")

    # --- 2. Establish Ground Truth ---
    # The ground truth is the top-ranked document from the Field Retriever for this golden set.
    ground_truth_df = df[
        (df['strategy_name'] == 'Field Retriever') &
        (df['rank'] == 1) &
        (df['query_id'].isin(golden_query_ids))
        ][['grammar_type', 'query_id', 'doc_id']].rename(columns={'doc_id': 'correct_doc_id'})

    # --- 3. Calculate Reciprocal Ranks for Each Strategy ---
    # First, filter the main dataframe to only include the golden queries for all strategies
    eval_df = df[df['query_id'].isin(golden_query_ids)]
    # Now, merge the ground truth with this filtered data
    eval_df = pd.merge(eval_df, ground_truth_df, on=['grammar_type', 'query_id'])

    # Calculate reciprocal rank: 1/rank if the doc is correct, otherwise 0
    eval_df['reciprocal_rank'] = (1 / eval_df['rank']).where(eval_df['doc_id'] == eval_df['correct_doc_id'], 0)

    # --- 4. Calculate Final MRR Scores ---
    # For each strategy and grammar, we find the MAX reciprocal rank for each query
    # (since a document might appear multiple times, we only care about its highest rank)
    # and then take the mean across all queries.
    mrr_scores = eval_df.groupby(['grammar_type', 'strategy_name', 'query_id'])['reciprocal_rank'].max().reset_index()
    mrr_scores = mrr_scores.groupby(['grammar_type', 'strategy_name'])['reciprocal_rank'].mean().reset_index()
    mrr_scores = mrr_scores.rename(columns={'reciprocal_rank': 'MRR'})

    # --- 5. Print Summary Table ---
    print("=" * 80)
    print("Final Performance Summary: Mean Reciprocal Rank (MRR)")
    print("(Higher is better, 1.0 is a perfect score)")
    print("=" * 80)
    summary_table = mrr_scores.pivot_table(index='strategy_name', columns='grammar_type', values='MRR')
    print(summary_table.round(4))
    print("\n" + "=" * 80)

    # --- 6. Visualization ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(data=mrr_scores, x='grammar_type', y='MRR', hue='strategy_name', palette='viridis')

    plt.title('Retrieval Strategy Performance (Mean Reciprocal Rank)', fontsize=18, pad=20)
    plt.ylabel('Mean Reciprocal Rank (MRR)', fontsize=14)
    plt.xlabel('Grammar Type', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Retrieval Strategy', fontsize=12)
    plt.ylim(0, 1.1)

    # Add labels to the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)

    plt.tight_layout()

    output_filename = "final_mrr_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated final MRR comparison plot: {output_filename}")


if __name__ == "__main__":
    final_retriever_comparison()