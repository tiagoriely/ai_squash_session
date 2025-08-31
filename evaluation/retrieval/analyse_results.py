# In evaluation/analyse_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# --- Make sure this utility can be imported ---
from rag.utils import load_and_format_config


def analyse_retriever_results(grammar_type: str):
    """
    Loads, normalises, and analyses retriever evaluation data, saving plots and stats.
    """
    # --- 1. Dynamically Build Paths ---
    project_root = Path(__file__).resolve().parent.parent.parent

    config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    base_config = load_and_format_config(str(config_path))
    corpus_size = base_config['corpus_size']

    results_dir = project_root / "evaluation" / "retrieval" / grammar_type / f"corpus_size_{corpus_size}"
    results_path = results_dir / f"retrieval_results_all-sets_{grammar_type}_{corpus_size}.csv"

    plots_dir = results_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- 2. Setup and Load Data ---
    sns.set_theme(style="whitegrid")

    try:
        df = pd.read_csv(results_path)
        print(f"✅ Data loaded successfully from: {results_path}")
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at '{results_path}'.")
        print(f"Please run the `evaluate_retrievers.py {grammar_type}` script first.")
        return

    # --- 3. Statistical Summary & Export ---
    print("\n📊" + "=" * 25 + " Statistical Summary (Raw Scores) " + "=" * 25)
    stats = df.groupby('retriever_name')['score'].describe()
    print(stats)

    # FIX: Export the statistical summary to a CSV file
    stats_path = plots_dir / f'statistical_summary_{grammar_type}.csv'
    stats.to_csv(stats_path)
    print(f"\n   -> Statistical summary saved to {stats_path}")

    # --- 4. Score Normalisation ---
    # We apply Min-Max scaling to each retriever's scores independently.
    # This scales all scores to a [0, 1] range for fair comparison in plots.
    print("\nNormalising scores for visualisation...")
    df['normalised_score'] = df.groupby('retriever_name')['score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    print("   -> Normalisation complete.")

    # --- 5. Visualisations (using normalised scores) ---

    # Plot 1: Normalised Score Distribution
    print("\nGenerating Plot 1: Normalised Score Distribution...")
    plt.figure(figsize=(12, 7))
    # FIX: Use the new 'normalised_score' column for the y-axis
    sns.boxplot(data=df, x='retriever_name', y='normalised_score', palette='viridis')
    plt.title(f'Normalised Score Distribution by Retriever ({grammar_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Retriever', fontsize=12)
    plt.ylabel('Normalised Score (0 to 1)', fontsize=12)
    plt.xticks(rotation=15)
    plot1_path = plots_dir / f'normalised_score_distribution_{grammar_type}.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot1_path}")
    plt.show()

    # Plot 2: Normalised Rank-Score Decay
    print("\nGenerating Plot 2: Normalised Rank-Score Decay...")
    plt.figure(figsize=(14, 8))
    # FIX: Use the new 'normalised_score' column for the y-axis
    sns.pointplot(data=df, x='rank', y='normalised_score', hue='retriever_name', palette='magma', errorbar='sd')
    plt.title(f'Average Normalised Score vs. Rank ({grammar_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Document Rank', fontsize=12)
    plt.ylabel('Average Normalised Score (with Std Dev)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Retriever', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot2_path = plots_dir / f'normalised_rank_score_decay_{grammar_type}.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot2_path}")
    plt.show()

    # Plot 3: Deeper Dive by Query Type with Normalised Scores
    print("\nGenerating Plot 3: Normalised Performance by Query Type...")
    # FIX: Use the new 'normalised_score' column for the y-axis
    g = sns.catplot(
        data=df, x='rank', y='normalised_score', hue='retriever_name', col='query_type',
        kind='point', palette='cividis', height=5, aspect=1.2, col_wrap=2, sharey=True
    )
    g.fig.suptitle(f'Normalised Rank-Score Decay by Query Type ({grammar_type})', y=1.03, fontsize=16,
                   fontweight='bold')
    g.set_axis_labels("Document Rank", "Average Normalised Score")
    g.set_titles("Query Type: {col_name}")
    g.tight_layout()
    plot3_path = plots_dir / f'normalised_decay_by_query_type_{grammar_type}.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot3_path}")
    plt.show()

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse retriever evaluation results for a specific grammar.")
    parser.add_argument(
        "grammar",
        type=str,
        choices=['balanced_grammar', 'high_constraint_grammar', 'loose_grammar'],
        help="The grammar type whose results you want to analyse."
    )
    args = parser.parse_args()

    analyse_retriever_results(grammar_type=args.grammar)