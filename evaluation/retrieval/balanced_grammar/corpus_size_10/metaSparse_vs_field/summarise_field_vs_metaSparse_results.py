import pandas as pd
import os


def analyse_and_summarise_csv(filepath):
    """
    Reads a CSV file, analyses it by grouping by 'query_type',
    calculates summary statistics, and returns a summary DataFrame.

    Args:
        filepath (str): The path to the input CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the summary statistics,
                          or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)

        # Automatically identify numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Group by 'query_type' and calculate summary statistics
        summary_df = df.groupby('query_type')[numeric_cols].agg(['mean', 'std', 'min', 'max'])

        # Flatten the column names for a cleaner CSV output (e.g., 'max_score_mean')
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]

        # Add a column to identify the source file
        summary_df['source_file'] = os.path.basename(filepath)

        return summary_df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        return None


if __name__ == '__main__':
    # --- Specify your files here ---
    files_to_analyse = [
        'field_retriever_metrics_more_queries_10.csv',
        'meta_sparse_metrics_more_queries_10.csv'
    ]

    # A list to hold the summary DataFrame from each file
    all_summaries = []

    # Loop through and process each file
    for file in files_to_analyse:
        print(f"Analysing {file}...")
        summary = analyse_and_summarise_csv(file)
        # Only add to the list if the file was processed successfully
        if summary is not None:
            all_summaries.append(summary)

    # Combine all summaries into a single DataFrame if any were successful
    if all_summaries:
        combined_summary_df = pd.concat(all_summaries)

        # Move the 'source_file' column to the front for better readability
        cols = ['source_file'] + [col for col in combined_summary_df.columns if col != 'source_file']
        combined_summary_df = combined_summary_df[cols]

        # Define the output filename and save the combined summary
        output_filename = 'combined_metrics_summary.csv'
        combined_summary_df.to_csv(output_filename, index=True)  # index=True to keep the 'query_type'

        print(f"\nAnalysis complete. Combined summary saved to '{output_filename}'")
    else:
        print("\nNo files were processed successfully. No output file was generated.")