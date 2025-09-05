import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyse_dense_performance():
    """
    Loads, analyses, and visualises the performance of the dense retriever.
    """
    try:
        df = pd.read_csv("dense_retriever_metrics.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'dense_retriever_metrics.csv' is in the correct directory.")
        return

    # --- 1. Data Preparation ---
    def standardise_query_type(q_type):
        if 'High-Relevance' in q_type:
            return 'High-Relevance'
        return q_type

    df['query_type_grouped'] = df['query_type'].apply(standardise_query_type)

    # --- 2. Quantitative Analysis ---
    overall_metrics = {
        'Avg. Max Score': df['max_score'].mean(),
        'Avg. Top-1 Delta': df['top_1_delta'].mean(),
        'Avg. Std Dev': df['std_dev'].mean()
    }

    category_metrics = df.groupby('query_type_grouped').agg(
        avg_max_score=('max_score', 'mean'),
        avg_top_1_delta=('top_1_delta', 'mean')
    ).reindex(['Complexity Type 1 (Relevant)', 'Complexity Type 2 (Relevant)',
               'Relevant (Other Duration)', 'Relevant (Single Shotside)', 'Vague But Relevant',
               'Relevant (Outside Corpus)', 'Out-of-Scope (Informational)',
               'Random (non-Relevant)'])

    # --- 3. Print Analysis Summary to Console ---
    print("=" * 80)
    print("Empirical Analysis: Dense (Semantic) Retriever Performance")
    print("=" * 80)
    print("\n### Overall Performance Metrics:\n")
    for key, value in overall_metrics.items():
        print(f"  - {key:<20}: {value:.4f}")
    print("\n" + "=" * 80 + "\n")
    print("### Performance by Query Category:\n")
    print(category_metrics.round(4))
    print("\n" + "=" * 80)

    # --- 4. Visualization ---
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    # FIX 1: Change subplot layout to a 2x2 grid and adjust figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Dense (Semantic) Retriever Performance Analysis', fontsize=18)
    # FIX 2: Create the 'ax_flat' variable by flattening the axes array
    ax_flat = axes.flatten()

    # Plot 1: Average Max Score by Query Type (using ax_flat[0])
    category_metrics['avg_max_score'].plot(
        kind='bar', ax=ax_flat[0], color='#fdc500'
    )
    ax_flat[0].set_title('Relevance by Query Type')
    ax_flat[0].set_xlabel('Query Category')
    ax_flat[0].set_ylabel('Average Max Score (Cosine Similarity)')
    plt.setp(ax_flat[0].get_xticklabels(), rotation=45, ha="right")
    ax_flat[0].bar_label(ax_flat[0].containers[0], fmt='%.3f', padding=3)

    # Plot 2: Distribution of All Max Scores (using ax_flat[1])
    sns.boxplot(y=df['max_score'], ax=ax_flat[1], color='#fdc500')
    ax_flat[1].set_title('Overall Score Distribution')
    ax_flat[1].set_ylabel('Max Score (Cosine Similarity)')

    # FIX 3: Moved the third plot's code here, before saving the figure
    # Plot 3: Score Progression for Graduated Complexity Queries
    complex_ids_1 = ["complex_01_cg", "complex_02_cg", "complex_03_cg"]
    complex_ids_2 = ["complex_21_mix", "complex_22_mix", "complex_23_mix", "complex_24_mix"]

    complex_queries1_df = df[df['query_id'].isin(complex_ids_1)].sort_values('query_id')
    complex_queries2_df = df[df['query_id'].isin(complex_ids_2)].sort_values('query_id')

    # Use the third axes object for the new plot (ax_flat[2])
    ax3 = ax_flat[2]
    ax3.plot(complex_queries1_df['query_id'], complex_queries1_df['max_score'], marker='o', linestyle='--',
             label='Complexity Type 1')
    ax3.plot(complex_queries2_df['query_id'], complex_queries2_df['max_score'], marker='o', linestyle='-',
             label='Complexity Type 2')
    ax3.set_xlabel('Query ID')
    ax3.set_ylabel('Max Score (Cosine Similarity)')
    ax3.set_title('Performance on Graduated Complexity Queries')
    ax3.legend()
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    # Hide the unused fourth subplot
    ax_flat[3].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "dense_retriever_analysis.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nGenerated analysis plots: {output_filename}")


if __name__ == "__main__":
    analyse_dense_performance()