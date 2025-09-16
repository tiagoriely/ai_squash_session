import pandas as pd

def analyse_retriever_metrics(input_filepath, output_filepath):
    """
    Analyses retriever metrics from a CSV file, grouping by query type
    and calculating summary statistics.

    Args:
        input_filepath (str): The path to the input CSV file.
        output_filepath (str): The path to save the summarised CSV file.
    """
    try:
        # Load the dataset from the specified file path
        df = pd.read_csv(input_filepath)

        # Define the numeric columns for which to calculate statistics
        numeric_cols = [
            'max_score', 'min_score', 'mean_score', 'std_dev',
            'q25', 'median', 'q75', 'top_1_delta'
        ]

        # Group the data by 'query_type' and calculate summary statistics
        # for the specified numeric columns.
        summary_df = df.groupby('query_type')[numeric_cols].agg(
            ['mean', 'std', 'min', 'max']
        )

        # Flatten the multi-level column headers for better readability in the CSV
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]

        # Save the resulting summary DataFrame to a new CSV file
        summary_df.to_csv(output_filepath)

        print(f"Analysis complete. Summary saved to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define the input and output file names
    input_file = 'dense_retriever_metrics_v1.csv'
    output_file = 'semantic_metrics_summary.csv'

    # Run the analysis function
    analyse_retriever_metrics(input_file, output_file)