# evaluation/evaluate_results.py

import json
import argparse
import yaml
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any


# Import metric functions and the LLM evaluator class
from metrics.structure_metrics import calculate_psa, calculate_pda, calculate_stc
from metrics.diversity_metrics import calculate_ipv, calculate_ipsd
from metrics.reliability_metrics import calculate_mas, calculate_pca
from evaluators.llm_judge import LlmEvaluator

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not found. IPSD metric will not be available.")
    SentenceTransformer = None

# Load environment
from dotenv import load_dotenv
load_dotenv()


def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL result file into a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_results(file_path: Path, results: List[Dict[str, Any]]):
    """Saves a list of dictionaries to a JSONL file."""

    # Get the parent directory of the output file
    output_dir = file_path.parent
    # Create the directory if it doesn't exist, including any parent folders.
    # exist_ok=True prevents an error if the directory already exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def group_results_by_query(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Groups results by question text for metrics like IPSD."""
    grouped = defaultdict(list)
    for res in results:
        grouped[res['question']].append(res)
    return grouped


def load_exercise_definitions(grammar_path: Path) -> Dict[str, Any]:
    """
    Loads all exercise variant definitions from the YAML files in a grammar directory.
    This creates a lookup table for metrics like PCA.
    """
    definitions = {}
    exercise_files = list(grammar_path.rglob("*.yaml"))
    for file_path in exercise_files:
        if "sessions_types" in file_path.name: continue  # Skip config files
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if 'variants' in data:
                for variant in data['variants']:
                    if 'variant_id' in variant:
                        definitions[variant['variant_id']] = variant
                        # Also map by name for easier lookup from generated text
                        if 'name' in variant:
                            definitions[variant['name']] = variant
    return definitions


def main(args):
    print("🚀 Starting Evaluation Pipeline...")

    # --- 1. INITIALIZATION ---
    print("   - Initializing models and loading resources...")
    project_root = Path(__file__).resolve().parent.parent

    # Initialize the LLM judge and embedding model
    llm_judge = LlmEvaluator(model_name="gpt-4-turbo-preview")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') if SentenceTransformer else None

    # Load all results from the experiment run
    results = load_results(args.input_file)
    print(f"   - Loaded {len(results)} results from {args.input_file.name}")

    # Load all exercise definitions from the source grammars
    all_exercise_definitions = load_exercise_definitions(args.grammar_dir)
    print(f"   - Loaded {len(all_exercise_definitions)} unique exercise definitions.")

    # Prepare grammar file paths for PSA metric
    grammar_to_ebnf = {
        "high_constraint_grammar": project_root / "grammar/dsl/strict_structures.ebnf",
        "balanced_grammar": project_root / "grammar/dsl/loose_structures.ebnf",
        "loose_grammar": project_root / "grammar/dsl/loose_structures.ebnf"
    }

    # Prepare config paths for PDA metric
    session_types_config_path = project_root / "grammar/sports/squash/sessions_types.yaml"
    # NOTE: Finding the exact source structure template is complex.
    # This is a placeholder assuming a location. You may need to adapt this logic.
    structure_template_dir = project_root / "grammar/sports/squash/high_constraint_grammar/_structures"

    # --- 2. MAIN EVALUATION LOOP ---
    print("   - Calculating metrics for each result...")
    for result in tqdm(results, desc="Evaluating Results"):
        # This dictionary will store all scores for the current result
        result['evaluation_scores'] = {}

        plan_text = result['answer']
        query = result['question']
        grammar_type = result['grammar_type']
        retrieved_docs_str = "\n---\n".join(result['contexts'])

        # --- Structure Metrics ---
        ebnf_path = grammar_to_ebnf.get(grammar_type)
        if ebnf_path and ebnf_path.exists():
            result['evaluation_scores']['psa_score'] = calculate_psa(plan_text, ebnf_path)

        # HACK: Infer structure template path based on duration. A more robust system would
        # log the source template during generation.
        duration_match = re.search(r"Duration:\s*(\d+)", plan_text)
        if duration_match:
            duration = int(duration_match.group(1))
            # Assuming a naming convention like 'structure_45min_...'. This will need adjustment.
            # For this example, we'll hardcode one to ensure the code runs.
            # In your real code, build a robust lookup.
            struct_path = structure_template_dir / "structure_45min_3blocks_cg.yaml"
            if struct_path.exists():
                result['evaluation_scores']['pda_score'] = calculate_pda(plan_text, struct_path,
                                                                         session_types_config_path)

        result['evaluation_scores']['stc_score'] = calculate_stc(plan_text, query)

        # --- Diversity Metrics ---
        result['evaluation_scores']['ipv_score'] = calculate_ipv(plan_text)

        # --- Reliability Metrics ---
        result['evaluation_scores']['mas_score'] = calculate_mas(plan_text, query)

        # --- LLM-as-Judge Metrics (can be slow and costly) ---
        lcs_eval = llm_judge.evaluate_logical_coherence(query=query, generated_plan=plan_text)
        result['evaluation_scores']['lcs_eval'] = lcs_eval  # Store full JSON for analysis

        faithfulness_eval = llm_judge.evaluate_faithfulness(context=retrieved_docs_str, generated_plan=plan_text)
        result['evaluation_scores']['faithfulness_eval'] = faithfulness_eval

        # This depends on the LLM judge and is part of reliability
        pca_score = calculate_pca(plan_text, result['retrieved_documents_info'], llm_judge, all_exercise_definitions)
        result['evaluation_scores']['pca_score'] = pca_score

    # --- 3. BATCH METRIC CALCULATION (IPSD) ---
    print("   - Calculating group-based metrics (IPSD)...")
    grouped_results = group_results_by_query(results)
    for query_group in grouped_results.values():
        if len(query_group) > 1 and embedding_model:
            plans = [res['answer'] for res in query_group]
            ipsd_score = calculate_ipsd(plans, embedding_model)
            # Assign the same score to all items in the group
            for res in query_group:
                if 'evaluation_scores' in res:
                    res['evaluation_scores']['ipsd_score'] = float(ipsd_score)

    # --- 4. SAVE ENRICHED RESULTS ---
    save_results(args.output_file, results)
    print(f"✅ Evaluation complete! Enriched results saved to {args.output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline on experiment results.")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the input JSONL file from an experiment run."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the output JSONL file with evaluation scores."
    )
    parser.add_argument(
        "--grammar-dir",
        type=Path,
        default="grammar/sports/squash/",
        help="Path to the root directory containing grammar YAML definitions."
    )

    cli_args = parser.parse_args()
    main(cli_args)