# scripts/compare_hybrid_retrievers_advanced.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalise_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the fusion_score for each retriever strategy to the [0, 1] range,
    except for 'dynamic_query_aware_score_fusion' which is already normalised.
    """
    strategies_to_normalise = [
        s for s in df['strategy_name'].unique()
        if s != 'dynamic_query_aware_score_fusion'
    ]
    df_normalised = df.copy()
    for strategy in strategies_to_normalise:
        strategy_mask = df_normalised['strategy_name'] == strategy
        scores = df_normalised.loc[strategy_mask, 'fusion_score']
        min_score = scores.min()
        max_score = scores.max()
        score_range = max_score - min_score
        if score_range > 0:
            df_normalised.loc[strategy_mask, 'fusion_score'] = (scores - min_score) / score_range
        else:
            df_normalised.loc[strategy_mask, 'fusion_score'] = 0.0
    return df_normalised


def create_advanced_plots():
    """
    Loads detailed results, selectively normalises scores, and creates
    two separate, advanced comparison plots.
    """
    input_path = "hybrid_retrievers_detailed_results.csv"
    try:
        df_original = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'analyse_hybrid_retrievers_advanced.py' first.")
        return

    # Create a globally normalised dataframe for the first plot
    df_norm = normalise_scores(df_original.copy())

    # --- Plot 1: Average Top-1 Delta (Confidence) ---
    print("Generating Plot 1: Retriever Confidence by Query Type...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    top_ranks = df_norm[df_norm['rank'].isin([1, 2])]
    deltas = top_ranks.pivot_table(index=['query_id', 'strategy_name'], columns='rank',
                                   values='fusion_score').reset_index()
    deltas.columns = ['query_id', 'strategy_name', 'rank1_score', 'rank2_score']
    deltas['rank2_score'] = deltas['rank2_score'].fillna(0)
    deltas['top_1_delta'] = deltas['rank1_score'] - deltas['rank2_score']

    def standardise_query_type(q_type):
        if 'Complexity' in q_type: return 'High-Relevance (Complex)'
        if 'Single Shotside' in q_type: return 'High-Relevance (Shotside)'
        if 'Vague' in q_type: return 'Vague But Relevant'
        if 'Other Sport' in q_type: return 'Out-of-Scope (Other Sport)'
        return 'Other'

    deltas = pd.merge(deltas, df_original[['query_id', 'query_type']].drop_duplicates(), on='query_id')
    deltas['query_type_grouped'] = deltas['query_type'].apply(standardise_query_type)
    avg_delta_by_type = deltas.groupby(['strategy_name', 'query_type_grouped'])['top_1_delta'].mean().unstack()

    avg_delta_by_type.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Retriever Confidence by Query Type', fontsize=16)
    ax1.set_ylabel('Average Top-1 Delta (Confidence)')
    ax1.set_xlabel('Hybrid Strategy')

    plt.setp(ax1.get_xticklabels(), rotation=25, ha='right')

    ax1.legend(title='Query Type')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    fig1.tight_layout()
    output_filename_1 = "retriever_confidence_by_type.png"
    fig1.savefig(output_filename_1, bbox_inches='tight')
    plt.close(fig1)
    print(f"-> Saved '{output_filename_1}'")

    # --- Plot 2: Normalised Score Drop-off (Ranking Quality) ---
    print("\nGenerating Plot 2: Average Score Drop-off...")
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    df_plot2 = df_original.copy()
    df_plot2['plot2_norm_score'] = df_plot2.groupby(['query_id', 'strategy_name'])['fusion_score'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] != 0 else 0)

    # Define custom styles for maximum clarity
    style_map = {
        'Static Weighted RRF': {
            "color": "#1f77b4", "marker": "o", "linestyle": "-"
        },
        'Standard Unweighted RRF': {
            "color": "#ff7f0e", "marker": "s", "linestyle": "-"
        },
        'Dynamic Query-Aware RRF (Rank)': {
            "color": "#2ca02c", "marker": "X", "linestyle": "-"
        },
        'Dynamic Query-Aware Fusion': {
            "color": "#d62728", "marker": "D", "linestyle": "-"
        }
    }

    # Loop through each strategy and plot it with its custom style
    for name, style in style_map.items():
        strategy_df = df_plot2[df_plot2['strategy_name'] == name]
        sns.lineplot(
            data=strategy_df,
            x='rank',
            y='plot2_norm_score',
            label=name,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=3,
            markersize=10,
            ax=ax2,
            errorbar='sd'
        )

    ax2.set_title('Average Score Drop-off (All Queries)', fontsize=16)
    ax2.set_ylabel('Normalised Fusion Score (Rank-1 = 1.0)')
    ax2.set_xlabel('Document Rank')
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend(title='Strategy Name')

    fig2.tight_layout()
    output_filename_2 = "plots/score_drop_off_comparison.png"
    fig2.savefig(output_filename_2, bbox_inches='tight')
    plt.close(fig2)
    print(f"-> Saved '{output_filename_2}'")


if __name__ == "__main__":
    create_advanced_plots()