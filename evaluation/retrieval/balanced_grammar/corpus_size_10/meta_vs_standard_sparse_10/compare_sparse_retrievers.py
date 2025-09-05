# scripts/compare_retrievers.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyse_and_plot_retriever_performance():
    """
    Loads, analyses, and visualises the performance comparison between
    the standard sparse and metadata sparse retrievers.
    """
    try:
        # Load the two CSV files into pandas DataFrames
        std_df = pd.read_csv("standard_sparse_metrics_10.csv")
        meta_df = pd.read_csv("metadata_sparse_metrics_10.csv")
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure both 'standard_sparse_metrics_10.csv' and 'metadata_sparse_metrics_10.csv' are in the same directory.")
        return

    # --- 1. Data Preparation ---
    # Merge the two dataframes for a direct, side-by-side comparison
    comparison_df = pd.merge(
        std_df,
        meta_df,
        on=['query_id', 'query_type', 'query_text'],
        suffixes=('_std', '_meta')
    )

    # --- 2. Quantitative Analysis ---
    # Calculate the key aggregate metrics for comparison
    avg_delta_std = comparison_df['top_1_delta_std'].mean()
    avg_delta_meta = comparison_df['top_1_delta_meta'].mean()

    high_relevance_df = comparison_df[comparison_df['query_type'] == 'High-Relevance']
    avg_max_score_std_hr = high_relevance_df['max_score_std'].mean()
    avg_max_score_meta_hr = high_relevance_df['max_score_meta'].mean()

    # Isolate the graduated complexity queries for the progression plot
    complex_queries = ["complex_01", "complex_02", "complex_03", "complex_04"]
    complex_df = comparison_df[comparison_df['query_id'].isin(complex_queries)].set_index('query_id').loc[
        complex_queries]

    # --- 3. Print Analysis Summary to Console ---
    print("=" * 80)
    print("Empirical Analysis: Standard Sparse vs. Metadata Sparse Retriever")
    print("=" * 80)
    print("\n### Key Metric Comparison:\n")
    print(f"  - Average Top-1 Delta (Confidence):")
    print(f"    - Standard Sparse: {avg_delta_std:.4f}")
    print(f"    - Metadata Sparse: {avg_delta_meta:.4f}\n")
    print(f"  - Average Max Score (High-Relevance Queries):")
    print(f"    - Standard Sparse: {avg_max_score_std_hr:.4f}")
    print(f"    - Metadata Sparse: {avg_max_score_meta_hr:.4f}\n")
    print("\n### Analysis of Graduated Complexity (complex_01 to complex_04):\n")
    print("Max Scores:")
    print(complex_df[['max_score_std', 'max_score_meta']])

    # --- 4. Visualization ---
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 6)  # Increased figure width for better spacing
    plt.rcParams['font.size'] = 12
    fig = plt.figure()

    # Plot 1: Average Top-1 Delta (Confidence)
    ax1 = fig.add_subplot(1, 3, 1)
    deltas = [avg_delta_std, avg_delta_meta]
    retrievers = ['Standard Sparse', 'Metadata Sparse']
    bars1 = ax1.bar(retrievers, deltas, color=['#ff9999', '#66b3ff'])
    ax1.set_ylabel('Score Difference (Rank 1 - Rank 2)')
    ax1.set_title('Average Top-1 Delta (Retriever Confidence)')
    ax1.bar_label(bars1, fmt='%.3f')

    # Plot 2: Average Max Score on High-Relevance Queries
    ax2 = fig.add_subplot(1, 3, 2)
    scores = [avg_max_score_std_hr, avg_max_score_meta_hr]
    bars2 = ax2.bar(retrievers, scores, color=['#ff9999', '#66b3ff'])
    ax2.set_ylabel('Average BM25 Score')
    ax2.set_title('Avg. Max Score on High-Relevance Queries')
    ax2.bar_label(bars2, fmt='%.3f')

    # Plot 3: Score Progression for Graduated Complexity Queries
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(complex_df.index, complex_df['max_score_std'], marker='o', linestyle='--', label='Standard Sparse')
    ax3.plot(complex_df.index, complex_df['max_score_meta'], marker='o', linestyle='-', label='Metadata Sparse')
    ax3.set_xlabel('Query ID')
    ax3.set_ylabel('Max BM25 Score')
    ax3.set_title('Performance on Graduated Complexity Queries')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.legend()

    plt.tight_layout()

    # --- 5. Save the Output ---
    output_filename = "retriever_comparison_plots.png"
    plt.savefig(output_filename)
    print(f"\nGenerated comparison plots: {output_filename}")


if __name__ == "__main__":
    analyse_and_plot_retriever_performance()