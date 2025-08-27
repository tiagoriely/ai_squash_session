import pandas as pd
import json
import argparse
from pathlib import Path


def analyze_metadata_proportions(file_path: Path):
    """
    Analyzes a JSON Lines file to calculate and print the proportions for ALL metadata fields.

    Args:
        file_path (Path): The path object pointing to the .jsonl file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{file_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        return

    # Create and normalize the DataFrame
    df = pd.DataFrame(data_list)
    meta_df = pd.json_normalize(df['meta'])

    print(f"--- Full Metadata Proportions Analysis for {file_path.name} ---")

    # Dynamically loop through every column found in the metadata
    for column in meta_df.columns:
        print("\n" + "=" * 50)
        print(f"📊 Analysis for: '{column}'")
        print("=" * 50)

        # Drop rows where the current column is empty
        series = meta_df[column].dropna()

        if series.empty:
            print("Skipping (no data available).")
            continue

        # --- Determine the type of analysis needed ---
        # Get the first item to check its type
        first_item = series.iloc[0]

        try:
            # Special handling for 'shotSide' which can have mixed string/list types
            if column == 'shotSide':
                proportions = series.apply(
                    lambda x: x if isinstance(x, list) else [x]
                ).explode().value_counts(normalize=True, ascending=False)

            # If the data is a list, explode it before counting
            elif isinstance(first_item, list):
                proportions = series.explode().value_counts(normalize=True, ascending=False)

            # For simple types (string, int, float), just count the values
            elif isinstance(first_item, (str, int, float)):
                proportions = series.value_counts(normalize=True, ascending=False)

            # If the type is complex (like a dict), skip it
            else:
                print(f"Skipping (complex data type: {type(first_item)}).")
                continue

            # Print the top 15 results for brevity
            print(proportions.head(15))

        except Exception as e:
            print(f"Could not analyze column '{column}'. Error: {e}")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze a .jsonl dataset of squash sessions and report all metadata proportions."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input .jsonl file."
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"❌ Error: Input file not found at '{args.input}'")
        return

    analyze_metadata_proportions(args.input)


if __name__ == '__main__':
    main()