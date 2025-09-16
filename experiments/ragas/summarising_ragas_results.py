import pandas as pd
import os


def analyse_ragas_scores():
    """
    Reads Ragas scores from a CSV, calculates summary statistics,
    and writes them to a new, properly formatted CSV file.

    The script calculates and outputs two separate tables:
    1. Overall average faithfulness and answer relevancy for each grammar type.
    2. Average scores for each grammar type, broken down by size.
    """
    # --- Configuration ---
    # Define the input and output file paths in-code
    input_path = 'ragas_scores_all_v1.csv'
    output_path = 'ragas_summary.csv'

    # --- Script Execution ---
    try:
        # Check if the input file exists before proceeding
        if not os.path.exists(input_path):
            print(f"❌ Error: The input file was not found at '{input_path}'")
            return

        # Read the source CSV file into a pandas DataFrame
        df = pd.read_csv(input_path)

        # --- 1. Calculate overall average score for each grammar type ---
        overall_summary = df.groupby('grammar')[['faithfulness', 'answer_relevancy']].mean().reset_index()

        # --- 2. Calculate average score for each grammar type per size ---
        per_size_summary = df.groupby(['grammar', 'size'])[['faithfulness', 'answer_relevancy']].mean().reset_index()
        per_size_summary = per_size_summary.sort_values(by=['grammar', 'size'])

        # --- 3. Write the summaries to a single, well-formatted CSV ---
        # Open the file in write mode ('w') to create it and add the first summary
        with open(output_path, 'w', newline='') as f:
            f.write("Overall Average Scores per Grammar Type\n")
            overall_summary.to_csv(f, index=False, float_format='%.4f')

            # Add two blank lines for clear visual separation
            f.write("\n\n")

        # Open the same file in append mode ('a') to add the second summary
        with open(output_path, 'a', newline='') as f:
            f.write("Average Scores per Grammar Type and Size\n")
            per_size_summary.to_csv(f, index=False, float_format='%.4f')

        print(f"✅ Summary report successfully generated at: '{output_path}'")
        print("The CSV file now contains two separate, correctly formatted tables.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to execute the script.
    """
    print("Starting the Ragas score analysis...")
    analyse_ragas_scores()
    print("Analysis complete.")


if __name__ == "__main__":
    main()