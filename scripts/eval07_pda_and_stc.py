# scripts/eval09_run_evaluation_from_file.py

import json
from pathlib import Path
import sys
import pandas as pd

# --- Environment and Path Setup ---
# Ensure the script can find project modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Evaluation Function Imports ---
# Make sure these paths are correct in your project structure
from evaluation.utils.context_metrics.structure_metrics import calculate_pda, calculate_stc
from rag.parsers.user_query_parser import parse_type

if __name__ == "__main__":
    # --- Input Configuration ---
    # Absolute path to the JSON file containing the generated sessions
    INPUT_FILE_PATH = Path("/archive/generated_sessions_12queries_set_k3_size100.json")

    # --- Metrics Configuration ---
    SESSION_TYPES_CONFIG_PATH = PROJECT_ROOT / "configs/session_types.yaml"

    # Define paths to the structure templates required for the PDA metric.
    # This dictionary maps [grammar][session_type] to the template directory.
    STRUCTURE_DIRECTORIES = {
        "high_constraint": {
            "conditioned_game": "grammar/sports/squash/high_constraint_grammar/session_structures/conditioned_games/",
            "mix": "grammar/sports/squash/high_constraint_grammar/session_structures/mix/",
            "drill": "grammar/sports/squash/high_constraint_grammar/session_structures/drills/"
        },
        "balanced": {
            "conditioned_game": "grammar/sports/squash/balanced_grammar/session_structures/conditioned_games/",
            "mix": "grammar/sports/squash/balanced_grammar/session_structures/mix/",
            "drill": "grammar/sports/squash/balanced_grammar/session_structures/drills/"
        },
        "loose": {
            # The loose grammar has a more general structure, primarily for 'mix' type sessions.
            "mix": "grammar/sports/squash/loose_grammar/session_structures/mix/"
        }
    }

    # --- 1. Load Generated Data ---
    try:
        print(f"Loading generated plans from: {INPUT_FILE_PATH}")
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        print(f"✅ Successfully loaded {len(generated_data)} items.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {INPUT_FILE_PATH}")
        print("Please ensure the path is correct and the generation script has been run.")
        sys.exit(1)

    evaluation_results = []

    # --- 2. Main Evaluation Loop ---
    print("\n--- Running evaluations on all generated plans ---")
    for item in generated_data:
        case_id = item['case_id']
        query_text = item['query_text']
        generated_plan = item['generated_plan']

        # --- Dynamically determine grammar and session type from the loaded data ---
        # The grammar is the first part of the case_id (e.g., 'balanced' from 'balanced_100_...')
        grammar = case_id.split('_')[0]
        # Parse the session type from the query text (e.g., 'conditioned game', 'mix')
        session_type = parse_type(query_text) or "mix"  # Default to 'mix' if not parsable

        # --- Find the correct structure template path for PDA calculation ---
        grammar_paths = STRUCTURE_DIRECTORIES.get(grammar, {})
        # Fallback to the 'mix' path if a specific session type path doesn't exist for that grammar
        template_dir_str = grammar_paths.get(session_type) or grammar_paths.get("mix")

        structure_template_path = None
        if template_dir_str:
            template_dir = PROJECT_ROOT / template_dir_str
            # Find the first YAML file in the directory to use as the template
            if template_dir.is_dir():
                yaml_files = sorted(list(template_dir.glob("*.yaml")))
                if yaml_files:
                    structure_template_path = yaml_files[0]

        # --- 3. Run Evaluation Metrics ---
        pda_score = 0.0
        # Only calculate PDA if a valid structure template was found
        if structure_template_path:
            pda_score = calculate_pda(generated_plan, structure_template_path, SESSION_TYPES_CONFIG_PATH)

        # Calculate semantic compliance between the query and the generated plan
        stc_score = calculate_stc(generated_plan, query_text)

        evaluation_results.append({
            'case_id': case_id,
            'pda_score': pda_score,
            'stc_score': stc_score
        })

    # --- 4. Display Final Summary ---
    print("\n" + "=" * 80)
    print("Final PDA & STC Evaluation Summary")
    print("=" * 80)

    if not evaluation_results:
        print("No results to display.")
    else:
        results_df = pd.DataFrame(evaluation_results)
        print(results_df.set_index('case_id').round(4))
        print("\n--- Overall Averages ---")
        # Exclude case_id from mean calculation and round the result
        print(results_df.drop(columns=['case_id']).mean().round(4))