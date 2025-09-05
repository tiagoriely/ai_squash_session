import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyse_and_plot_retriever_performance():
    """
    Loads, analyses, and visualises the performance comparison between
    the Field Retriever and Metadata Sparse retriever.
    """
    try:
        # Load the two CSV files into pandas DataFrames
        field_df = pd.read_csv("field_retriever_metrics_more_queries_10.csv")
        meta_df = pd.read_csv("meta_sparse_metrics_more_queries_10.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both CSV files are in the correct directory.")
        return

    # --- 1. Data Preparation ---
    def standardise_query_type(q_type):
        if 'High-Relevance' in q_type:
            return 'High-Relevance'
        return q_type

    field_df['query_type_grouped'] = field_df['query_type'].apply(standardise_query_type)
    meta_df['query_type_grouped'] = meta_df['query_type'].apply(standardise_query_type)

    comparison_df = pd.merge(
        field_df,
        meta_df,
        on=['query_id', 'query_type', 'query_text', 'query_type_grouped'],
        suffixes=('_field', '_meta')
    )

    # --- 2. Quantitative Analysis ---
    print("="*80)
    print("Final Empirical Analysis: Field Retriever vs. Metadata Sparse Retriever (28 Queries)")
    print("="*80)
    print("\n### Overall Performance Metrics:\n")
    overall_metrics_df = pd.DataFrame({
        "Avg. Top-1 Delta (Confidence)": {
            "Field Retriever": comparison_df['top_1_delta_field'].mean(),
            "Metadata Sparse": comparison_df['top_1_delta_meta'].mean()
        },
        "Avg. Max Score": {
            "Field Retriever": comparison_df['max_score_field'].mean(),
            "Metadata Sparse": comparison_df['max_score_meta'].mean()
        }
    })
    print(overall_metrics_df)
    print("\n" + "="*80)

    print("\n### Average Max Score by Query Category:\n")
    avg_scores_by_category = comparison_df.groupby('query_type_grouped').agg({
        'max_score_field': 'mean',
        'max_score_meta': 'mean'
    }).rename(columns={'max_score_field': 'avg_max_score_field', 'max_score_meta': 'avg_max_score_meta'})
    print(avg_scores_by_category)
    print("\n" + "="*80)


    # --- 3. Visualisation ---
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle('Field Retriever vs. Metadata Sparse Retriever Performance', fontsize=16)

    # Plot 1: Average Max Score by Query Type
    ax1 = fig.add_subplot(2, 2, 1)
    avg_score_plot = comparison_df.groupby('query_type_grouped')[['max_score_field', 'max_score_meta']].mean()
    avg_score_plot.plot(kind='bar', ax=ax1, color=['#0077b6', '#fca311'])
    ax1.set_title('Average Max Score by Query Type')
    ax1.set_ylabel('Average Max Score')
    ax1.set_xlabel('Query Type')
    ax1.legend(['Field Retriever', 'Metadata Sparse'])
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")


    # Plot 2: Average Top-1 Delta by Query Type
    ax2 = fig.add_subplot(2, 2, 2)
    avg_delta_plot = comparison_df.groupby('query_type_grouped')[['top_1_delta_field', 'top_1_delta_meta']].mean()
    avg_delta_plot.plot(kind='bar', ax=ax2, color=['#0077b6', '#fca311'])
    ax2.set_title('Average Top-1 Delta (Confidence) by Query Type')
    ax2.set_ylabel('Average Top-1 Delta')
    ax2.set_xlabel('Query Type')
    ax2.legend(['Field Retriever', 'Metadata Sparse'])
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Plot 3: Distribution of Max Scores
    ax3 = fig.add_subplot(2, 2, 3)
    score_dist_df = comparison_df[['max_score_field', 'max_score_meta']].melt(
        var_name='Retriever', value_name='Max Score'
    )
    score_dist_df['Retriever'] = score_dist_df['Retriever'].map({
        'max_score_field': 'Field Retriever', 'max_score_meta': 'Metadata Sparse'
    })
    sns.boxplot(x='Retriever', y='Max Score', data=score_dist_df, ax=ax3, palette=['#0077b6', '#fca311'])
    ax3.set_title('Overall Score Distribution')
    ax3.set_xlabel('')
    ax3.set_ylabel('Max Score')

    # Filter for complexity queries to be used in the 4th plot
    complex_df = comparison_df[comparison_df['query_id'].str.startswith('complex_')].sort_values('query_id')

    # Plot 4: Score Progression for Graduated Complexity Queries
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(complex_df['query_id'], complex_df['max_score_field'], marker='o', linestyle='--', label='Field Retriever')
    ax4.plot(complex_df['query_id'], complex_df['max_score_meta'], marker='o', linestyle='-', label='Metadata Sparse')
    ax4.set_xlabel('Query ID')
    ax4.set_ylabel('Max BM25 Score')
    ax4.set_title('Performance on Graduated Complexity Queries')
    ax4.legend()
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('retriever_performance_comparison.png')
    print("\nPlot saved as retriever_performance_comparison.png")

if __name__ == '__main__':
    analyse_and_plot_retriever_performance()