# scripts/analyse_hybrid_by_grammar.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyse_dynamic_hybrid_by_grammar():
    """
    Loads the consolidated results, isolates the Dynamic Hybrid RRF strategy,
    and computes detailed performance statistics for each grammar type.
    """
    input_path = "final_evaluation_all_grammars_size100.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'run_final_evaluation.py' first.")
        return

    # --- 1. Filter for the Dynamic Hybrid Retriever ---
    hybrid_df = df[df['strategy_name'] == 'Dynamic Hybrid RRF'].copy()

    # --- 2. Calculate Top-1 Delta for each query and grammar ---
    top_ranks = hybrid_df[hybrid_df['rank'].isin([1, 2])]
    deltas = top_ranks.pivot_table(
        index=['grammar_type', 'query_id'],
        columns='rank',
        values='score'
    ).reset_index()
    deltas.columns = ['grammar_type', 'query_id', 'rank1_score', 'rank2_score']
    deltas['top_1_delta'] = deltas['rank1_score'] - deltas['rank2_score']

    # Calculate the average delta for each grammar
    avg_deltas = deltas.groupby('grammar_type')['top_1_delta'].mean().reset_index()

    # --- 3. Calculate Descriptive Statistics for all scores ---
    # The .describe() method efficiently calculates mean, std, and quartiles (25%, 50%, 75%)
    desc_stats = hybrid_df.groupby('grammar_type')['score'].describe()

    # --- 4. Calculate Average Rank-1 Score ---
    avg_rank1_scores = hybrid_df[hybrid_df['rank'] == 1].groupby('grammar_type')['score'].mean().reset_index()
    avg_rank1_scores = avg_rank1_scores.rename(columns={'score': 'avg_rank1_score'})

    # --- 5. Combine all metrics into a final summary table ---
    summary_df = pd.merge(desc_stats, avg_deltas, on='grammar_type')
    summary_df = pd.merge(summary_df, avg_rank1_scores, on='grammar_type')

    # Select and reorder columns for clarity
    final_cols = [
        'grammar_type', 'avg_rank1_score', 'top_1_delta',
        'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    ]
    summary_df = summary_df[final_cols].rename(columns={'50%': 'median'})

    # --- 6. Print and Save the Results ---
    print("=" * 80)
    print("Final Performance Statistics for Dynamic Hybrid RRF by Grammar")
    print("=" * 80)
    print(summary_df.round(4).to_string(index=False))

    output_path = PROJECT_ROOT / "dynamic_hybrid_grammar_comparison_stats.csv"
    summary_df.to_csv(output_path, index=False, float_format='%.7f')
    print(f"\n\n✅ Detailed statistics saved to: {output_path}")


if __name__ == "__main__":
    analyse_dynamic_hybrid_by_grammar()