# scripts/final_dropoff_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_final_dropoff_plots():
    """
    Loads the consolidated results and generates a focused, 3-panel plot
    showing the normalised score drop-off for each grammar type.
    """
    input_path = "../../../archive/final_evaluation_all_grammars_size100.csv"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please run 'run_final_evaluation.py' first.")
        return

    # --- 1. Data Preparation ---
    # Normalise scores across the entire dataframe to make curves comparable
    # This divides every score by the rank-1 score for that specific query/strategy/grammar group
    df['normalised_score'] = df.groupby(['grammar_type', 'query_id', 'strategy_name'])['score'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] != 0 else 0
    )

    # --- 2. Visualization ---
    sns.set_style("whitegrid")

    # Define the order of grammars for the subplots
    grammar_order = ['loose', 'balanced', 'high_constraint']

    # Create a 1x3 figure for the three grammar types
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.suptitle('Retrieval Strategy Ranking Quality Across Different Grammars', fontsize=22, y=1.03)

    for i, grammar in enumerate(grammar_order):
        ax = axes[i]
        # Filter the data for the current grammar
        grammar_df = df[df['grammar_type'] == grammar]

        # Create the line plot showing the score drop-off
        sns.lineplot(
            data=grammar_df,
            x='rank',
            y='normalised_score',
            hue='strategy_name',
            style='strategy_name',
            markers=True,
            dashes=False,
            ax=ax,
            palette='viridis',
            errorbar='sd'  # Show standard deviation as a shaded area
        )

        ax.set_title(f'Grammar: {grammar.replace("_", " ").title()}', fontsize=18)
        ax.set_ylabel('Normalised Score (Rank-1 = 1.0)' if i == 0 else '')
        ax.set_xlabel('Document Rank')
        ax.set_xticks(range(1, 11))
        ax.grid(True, which='both', linestyle='--')

        # --- The "Zoom-In" ---
        ax.set_ylim(0.5, 1.15)

        # Tidy up legends
        if i < 2:
            ax.get_legend().remove()
        else:
            ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "../../../archive/final_dropoff_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated final drop-off comparison plots: {output_filename}")


if __name__ == "__main__":
    create_final_dropoff_plots()