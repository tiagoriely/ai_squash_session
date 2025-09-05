# scripts/compare_hybrid_retrievers_advanced.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_advanced_plots():
    """
    Loads the detailed hybrid results and creates advanced comparison plots.
    """
    input_path = "05_hybrid_retrievers_detailed_results.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'analyse_hybrid_retrievers_advanced.py' first.")
        return

    # --- Plot 1: Average Top-1 Delta by Query Type (Confidence) ---
    top_ranks = df[df['rank'].isin([1, 2])]
    deltas = top_ranks.pivot_table(index=['query_id', 'strategy_name'], columns='rank',
                                   values='fusion_score').reset_index()
    deltas.columns = ['query_id', 'strategy_name', 'rank1_score', 'rank2_score']
    deltas['top_1_delta'] = deltas['rank1_score'] - deltas['rank2_score']

    # Standardise query types
    def standardise_query_type(q_type):
        if 'Complexity' in q_type: return 'High-Relevance (Complex)'
        if 'Single Shotside' in q_type: return 'High-Relevance (Shotside)'
        if 'Vague' in q_type: return 'Vague But Relevant'
        if 'Other Sport' in q_type: return 'Out-of-Scope (Other Sport)'
        return 'Other'

    deltas = pd.merge(deltas, df[['query_id', 'query_type']].drop_duplicates(), on='query_id')
    deltas['query_type_grouped'] = deltas['query_type'].apply(standardise_query_type)

    avg_delta_by_type = deltas.groupby(['strategy_name', 'query_type_grouped'])['top_1_delta'].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Advanced Hybrid Retriever Comparison', fontsize=18)

    avg_delta_by_type.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('Retriever Confidence by Query Type')
    axes[0].set_ylabel('Average Top-1 Delta (Confidence)')
    axes[0].set_xlabel('Hybrid Strategy')
    axes[0].tick_params(axis='x', rotation=45)

    # --- Plot 2: Normalised Score Drop-off (Ranking Quality) ---
    # Normalise scores (divide by the rank-1 score for each query/strategy group)
    df['normalised_score'] = df.groupby(['query_id', 'strategy_name'])['fusion_score'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] != 0 else 0)

    sns.lineplot(data=df, x='rank', y='normalised_score', hue='strategy_name', marker='o', ax=axes[1], errorbar='sd')
    axes[1].set_title('Average Score Drop-off (All Queries)')
    axes[1].set_ylabel('Normalised Fusion Score (Rank-1 = 1.0)')
    axes[1].set_xlabel('Document Rank')
    axes[1].set_xticks(range(1, 11))
    axes[1].grid(True, which='both', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = "hybrid_retriever_advanced_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated advanced comparison plots: {output_filename}")


if __name__ == "__main__":
    create_advanced_plots()