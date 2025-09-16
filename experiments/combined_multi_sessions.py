import json
from pathlib import Path

# --- Configuration ---
# The directory where your generated JSON files are located.
# This assumes your script is in the root of your project.
INPUT_DIR = Path("experiments")

# The pattern to match. This will find all files in the INPUT_DIR
# that start with "evaluation_sessions_" and end with ".json".
# Adjust this if your filenames are different.
GLOB_PATTERN = "evaluation_sessions_multi_*.json"

# The name of the final, combined output file.
OUTPUT_FILENAME = INPUT_DIR / "master_evaluation_file.json"


# --- End of Configuration ---

def combine_json_files():
    """
    Finds all JSON files matching a pattern in a directory,
    combines them, and saves them to a new master file.
    """
    all_results = []

    # Find all files matching the pattern
    files_to_merge = sorted(list(INPUT_DIR.glob(GLOB_PATTERN)))

    if not files_to_merge:
        print(f"❌ Error: No files found in '{INPUT_DIR}' matching the pattern '{GLOB_PATTERN}'.")
        print("Please check the INPUT_DIR and GLOB_PATTERN variables in the script.")
        return

    print(f"Found {len(files_to_merge)} files to merge...")

    # Loop through each file, read its content, and append to the master list
    for file_path in files_to_merge:
        print(f"  -> Reading {file_path.name}")
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_results.extend(data)
            else:
                print(f"     - Warning: File {file_path.name} does not contain a JSON list. Skipping.")
        except json.JSONDecodeError:
            print(f"     - Warning: Could not decode JSON from {file_path.name}. Skipping.")

    # Write the combined list to the new master file
    if all_results:
        print(f"\nWriting a total of {len(all_results)} records to the master file...")
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Success! Master file created at: {OUTPUT_FILENAME}")
    else:
        print("No data was merged.")


if __name__ == "__main__":
    combine_json_files()