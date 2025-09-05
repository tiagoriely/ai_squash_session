# scripts/compare_all_strategies.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyse_final_results():
    """
    Loads the consolidated results and generates final comparison plots for
    all retrievers across all grammar types.
    """
    input_path = "../evaluation/retrieval/all_corpora/final_evaluation_all_grammars_size100.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'run_final_evaluation.py' first.")
        return

    # --- 1. Data Preparation and Metric Calculation ---

    # Calculate Top-1 Delta for each query group
    top_ranks = df[df['rank'].isin([1, 2])]
    deltas = top_ranks.pivot_table(
        index=['grammar_type', 'query_id', 'strategy_name'],
        columns='rank',
        values='score'
    ).reset_index()
    deltas.columns = ['grammar_type', 'query_id', 'strategy_name', 'rank1_score', 'rank2_score']
    deltas['top_1_delta'] = deltas['rank1_score'] - deltas['rank2_score']

    # Calculate average metrics for plotting
    avg_metrics = deltas.groupby(['grammar_type', 'strategy_name']).agg(
        avg_top_1_delta=('top_1_delta', 'mean')
    ).reset_index()

    # Get average max score (rank 1 score)
    avg_max_score = df[df['rank'] == 1].groupby(['grammar_type', 'strategy_name'])['score'].mean().reset_index()
    avg_metrics = pd.merge(avg_metrics, avg_max_score.rename(columns={'score': 'avg_max_score'}),
                           on=['grammar_type', 'strategy_name'])

    # --- 2. Print Summary Table ---
    print("=" * 80)
    print("Final Performance Summary Across All Grammars")
    print("=" * 80)
    summary_table = avg_metrics.pivot_table(index='strategy_name', columns='grammar_type',
                                            values=['avg_max_score', 'avg_top_1_delta'])
    print(summary_table.round(3))
    print("\n" + "=" * 80)

    # --- 3. Visualization ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
    fig.suptitle('Retrieval Strategy Performance Across Grammars (Corpus Size 100)', fontsize=22, y=1.02)

    # Plot 1: Overall Relevance (Average Max Score)
    sns.barplot(data=avg_metrics, x='grammar_type', y='avg_max_score', hue='strategy_name', ax=axes[0],
                palette='viridis')
    axes[0].set_title('Overall Relevance')
    axes[0].set_ylabel('Average Top-1 Score')
    axes[0].set_xlabel('Grammar Type')
    axes[0].legend(title='Strategy')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Overall Confidence (Average Top-1 Delta)
    sns.barplot(data=avg_metrics, x='grammar_type', y='avg_top_1_delta', hue='strategy_name', ax=axes[1],
                palette='viridis')
    axes[1].set_title('Overall Confidence')
    axes[1].set_ylabel('Average Top-1 Delta (R1-R2 Score)')
    axes[1].set_xlabel('Grammar Type')
    axes[1].legend(title='Strategy')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Score Drop-off Curves (Ranking Quality) for the Balanced Grammar
    balanced_df = df[df['grammar_type'] == 'balanced'].copy()

    # Normalise scores to make curves comparable
    # This line will now operate on a guaranteed copy, silencing the warning.
    balanced_df['normalised_score'] = balanced_df.groupby(['query_id', 'strategy_name'])['score'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] != 0 else 0)

    sns.lineplot(data=balanced_df, x='rank', y='normalised_score', hue='strategy_name', style='strategy_name',
                 markers=True, dashes=False, ax=axes[2], palette='viridis', errorbar='sd')
    axes[2].set_title('Score Drop-off (Balanced Grammar)')

    sns.lineplot(data=balanced_df, x='rank', y='normalised_score', hue='strategy_name', style='strategy_name',
                 markers=True, dashes=False, ax=axes[2], palette='viridis', errorbar='sd')
    axes[2].set_title('Score Drop-off (Balanced Grammar)')
    axes[2].set_ylabel('Normalised Score (Rank-1 = 1.0)')
    axes[2].set_xlabel('Document Rank')
    axes[2].set_xticks(range(1, 11))
    axes[2].grid(True, which='both', linestyle='--')
    axes[2].legend(title='Strategy')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "final_evaluation_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated final comparison plots: {output_filename}")


if __name__ == "__main__":
    analyse_final_results()