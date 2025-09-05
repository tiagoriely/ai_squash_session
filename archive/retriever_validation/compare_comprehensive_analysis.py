# scripts/final_comprehensive_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def final_comprehensive_analysis():
    """
    Loads the consolidated results and generates a final, comprehensive
    set of comparison plots for all strategies and grammars.
    """
    input_path = "../evaluation/retrieval/all_corpora/final_evaluation_all_grammars_size100.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'run_final_evaluation.py' first.")
        return

    # --- 1. Data Preparation ---
    # Define the "Golden Set" for the precision (MRR) calculation
    golden_query_ids = [
        "complex_01_cg", "complex_02_cg", "complex_03_cg",
        "complex_21_mix", "complex_22_mix", "complex_23_mix", "complex_24_mix",
        "shotside_01", "shotside_02"
    ]

    # --- 2. Calculate Metrics for Each Plot ---

    # Metric for Plot 1: MRR on the Golden Set
    ground_truth_df = \
    df[(df['strategy_name'] == 'Field Retriever') & (df['rank'] == 1) & (df['query_id'].isin(golden_query_ids))][
        ['grammar_type', 'query_id', 'doc_id']].rename(columns={'doc_id': 'correct_doc_id'})
    mrr_eval_df = df[df['query_id'].isin(golden_query_ids)]
    mrr_eval_df = pd.merge(mrr_eval_df, ground_truth_df, on=['grammar_type', 'query_id'])
    mrr_eval_df['reciprocal_rank'] = (1 / mrr_eval_df['rank']).where(
        mrr_eval_df['doc_id'] == mrr_eval_df['correct_doc_id'], 0)
    mrr_scores = mrr_eval_df.groupby(['grammar_type', 'strategy_name'])['reciprocal_rank'].mean().reset_index().rename(
        columns={'reciprocal_rank': 'MRR'})

    # Metric for Plot 2: Avg. Max Score on Vague Queries
    vague_df = df[(df['query_type'] == 'Vague But Relevant') & (df['rank'] == 1)]
    vague_scores = vague_df.groupby(['grammar_type', 'strategy_name'])['score'].mean().reset_index().rename(
        columns={'score': 'Avg. Max Score'})

    # Data for Plot 3: Normalised Score Drop-off for Balanced Grammar
    balanced_df = df[df['grammar_type'] == 'balanced'].copy()
    # Normalise scores to make curves comparable
    balanced_df['normalised_score'] = balanced_df.groupby(['query_id', 'strategy_name'])['score'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] != 0 else 0)

    # --- 3. Visualization ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Comprehensive Retriever Performance Analysis', fontsize=22, y=1.02)

    # Plot 1: Precision (MRR on High-Relevance Queries)
    sns.barplot(data=mrr_scores, x='grammar_type', y='MRR', hue='strategy_name', ax=axes[0], palette='viridis')
    axes[0].set_title('Precision on Specific Queries (MRR)')
    axes[0].set_ylabel('Mean Reciprocal Rank (Higher is Better)')
    axes[0].set_xlabel('Grammar Type')
    axes[0].legend(title='Strategy')
    axes[0].set_ylim(0, 1.1)

    # Plot 2: Flexibility (Performance on Vague Queries)
    sns.barplot(data=vague_scores, x='grammar_type', y='Avg. Max Score', hue='strategy_name', ax=axes[1],
                palette='viridis')
    axes[1].set_title('Flexibility on Vague Queries')
    axes[1].set_ylabel('Average Top-1 Score')
    axes[1].set_xlabel('Grammar Type')
    axes[1].legend(title='Strategy')

    # Plot 3: Ranking Quality (Score Drop-off for Balanced Grammar)
    sns.lineplot(data=balanced_df, x='rank', y='normalised_score', hue='strategy_name', style='strategy_name',
                 markers=True, dashes=False, ax=axes[2], palette='viridis', errorbar='sd')
    axes[2].set_title('Ranking Quality (Balanced Grammar)')
    axes[2].set_ylabel('Normalised Score (Rank-1 = 1.0)')
    axes[2].set_xlabel('Document Rank')
    axes[2].set_xticks(range(1, 11))
    axes[2].grid(True, which='both', linestyle='--')
    axes[2].legend(title='Strategy')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = "final_comprehensive_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated final comprehensive comparison plots: {output_filename}")


if __name__ == "__main__":
    final_comprehensive_analysis()