# In evaluation/analyse_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# --- Make sure this utility can be imported ---
# You might need to adjust your PYTHONPATH if this fails
from rag.utils import load_and_format_config


def analyse_retriever_results(grammar_type: str):
    """
    Loads retriever evaluation data for a specific grammar and generates analysis plots.
    """
    # --- 1. Dynamically Build Paths ---
    project_root = Path(__file__).resolve().parent.parent.parent

    # Load a config to get the corpus_size dynamically
    # This ensures consistency with the evaluation script
    config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    base_config = load_and_format_config(str(config_path))
    corpus_size = base_config['corpus_size']

    # Construct the path to the specific results file
    results_dir = project_root / "evaluation" / "retrieval" / grammar_type / f"corpus_size_{corpus_size}"
    results_path = results_dir / f"retrieval_results_all-sets_{grammar_type}_{corpus_size}.csv"

    # Create a dedicated directory for the plots for this run
    plots_dir = results_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- 2. Setup and Load Data ---
    sns.set_theme(style="whitegrid")

    try:
        df = pd.read_csv(results_path)
        print(f"✅ Data loaded successfully from: {results_path}")
        print(f"Shape of the dataframe: {df.shape}\n")
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at '{results_path}'.")
        print(f"Please run the `evaluate_retrievers.py {grammar_type}` script first.")
        return

    # --- 3. Statistical Summary ---
    print("📊" + "=" * 25 + " Statistical Summary " + "=" * 25)
    stats = df.groupby('retriever_name')['score'].describe()
    print(stats)
    print("\n💡 Key metrics: High 'std' (standard deviation) suggests better discriminatory power.\n")

    # --- 4. Visualizations (saving with dynamic names) ---

    # Plot 1: Score Distribution
    print("Generating Plot 1: Score Distribution...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='retriever_name', y='score', palette='viridis')
    plt.title(f'Score Distribution by Retriever ({grammar_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Retriever', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=15)
    plot1_path = plots_dir / f'score_distribution_{grammar_type}.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot1_path}")
    plt.show()

    # Plot 2: Rank-Score Decay
    print("\nGenerating Plot 2: Rank-Score Decay...")
    plt.figure(figsize=(14, 8))
    sns.pointplot(data=df, x='rank', y='score', hue='retriever_name', palette='magma', errorbar='sd')
    plt.title(f'Average Score vs. Rank ({grammar_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Document Rank', fontsize=12)
    plt.ylabel('Average Score (with Std Dev)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Retriever', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot2_path = plots_dir / f'rank_score_decay_{grammar_type}.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot2_path}")
    plt.show()

    # Plot 3: Deeper Dive by Query Type
    print("\nGenerating Plot 3: Performance by Query Type...")
    g = sns.catplot(
        data=df, x='rank', y='score', hue='retriever_name', col='query_type',
        kind='point', palette='cividis', height=5, aspect=1.2
    )
    g.fig.suptitle(f'Rank-Score Decay by Query Type ({grammar_type})', y=1.03, fontsize=16, fontweight='bold')
    g.set_axis_labels("Document Rank", "Average Score")
    g.set_titles("Query Type: {col_name}")
    g.tight_layout()
    plot3_path = plots_dir / f'decay_by_query_type_{grammar_type}.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved plot to {plot3_path}")
    plt.show()

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    # --- Add command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Analyse retriever evaluation results for a specific grammar.")
    parser.add_argument(
        "grammar",
        type=str,
        choices=['balanced_grammar', 'high_constraint_grammar', 'loose_grammar'],
        help="The grammar type whose results you want to analyse."
    )
    args = parser.parse_args()

    analyse_retriever_results(grammar_type=args.grammar)