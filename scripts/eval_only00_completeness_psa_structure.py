# scripts/eval_completeness_gain_from_file.py

import yaml
import json
from pathlib import Path
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# --- Core Component Imports ---
from rag.generation.generator import Generator
from evaluation.utils.completeness_gain import CompletenessGain
from evaluation.utils.context_metrics.structure_metrics import calculate_psa_flexible, check_complex_structure

if __name__ == "__main__":
    # --- Input Configuration ---
    # Absolute path to the JSON file with the RAG-generated plans
    INPUT_FILE_PATH = "/Users/tiago/projects/ai_squash_session/evaluation_sessions_set_k3_size1000_20250907_145234.json"

    # --- Evaluation Parameters ---
    # Use zero temperature for deterministic baseline generation
    GENERATION_TEMPERATURE = 0.0
    # Paths to the EBNF grammar files needed for CG and PSA metrics
    GRAMMAR_PATHS = {
        "loose": "grammar/sports/squash/loose_grammar/loose_structures.ebnf",
        "balanced": "grammar/sports/squash/balanced_grammar/loose_structures.ebnf",
        "high_constraint": "grammar/sports/squash/high_constraint_grammar/strict_structures.ebnf"
    }

    # --- 1. Initialise Required Components ---
    # Generator is needed to create the "baseline" plan for comparison
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. It's needed to generate baseline plans.")
    llm_generator = Generator(model="gpt-4o")

    # Pre-initialise an evaluator for each grammar type for efficiency
    print("--- Initialising CompletenessGain evaluators for all grammar types ---")
    completeness_evaluators = {
        grammar: CompletenessGain(ebnf_grammar_path=PROJECT_ROOT / path)
        for grammar, path in GRAMMAR_PATHS.items()
    }
    print("✅ Evaluators and Generator initialised.")

    # --- 2. Load Pre-Generated Data (Target Plans) ---
    try:
        print(f"\nLoading RAG-generated 'target' plans from: {INPUT_FILE_PATH}")
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        print(f"✅ Successfully loaded {len(generated_data)} items.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {INPUT_FILE_PATH}")
        sys.exit(1)

    evaluation_results = []

    # --- 3. Main Evaluation Loop ---
    # This prompt template will be used to generate the baseline (non-RAG) plan
    prompt_template = """You are an expert squash coach AI. Your task is to generate a detailed and coherent squash session plan.
    The user's request is: "{query}".
    The final output must be a single, complete, and well-structured session plan.

    CONTEXT:
    {context}

    FINAL SQUASH SESSION PLAN:"""

    print("\n--- Starting Evaluation Loop ---")
    for item in generated_data:
        case_id = item['case_id']
        query_text = item['query_text']
        # The plan from the file is our "target" (RAG-enhanced) plan
        target_plan = item['generated_plan']

        print("\n" + "=" * 80)
        print(f"Processing case: {case_id}")

        # --- Dynamically get grammar config for the current item ---
        grammar = case_id.split('_')[0]
        if grammar not in GRAMMAR_PATHS:
            print(f"  -> ⚠️ SKIPPING: Grammar '{grammar}' not found in GRAMMAR_PATHS config.")
            continue

        ebnf_grammar_path = PROJECT_ROOT / GRAMMAR_PATHS[grammar]
        completeness_evaluator = completeness_evaluators[grammar]

        # --- STAGE 1: Generate BASELINE PLAN (without RAG context) ---
        print("  -> Generating BASELINE plan (without RAG context)...")
        baseline_context = "No external context provided. Generate a plan based on general knowledge."
        baseline_prompt = prompt_template.format(query=query_text, context=baseline_context)
        baseline_plan = llm_generator.generate(baseline_prompt, temperature=GENERATION_TEMPERATURE)

        # --- STAGE 2: Run All Evaluations ---
        print("  -> Evaluating target vs. baseline...")
        # A) Completeness Gain
        cg_scores = completeness_evaluator.get_score(generated_plan=target_plan, baseline_text=baseline_plan)
        # B) Flexible Plan Structure Adherence (PSA)
        psa_resp = calculate_psa_flexible(target_plan, ebnf_grammar_path)
        psa_base = calculate_psa_flexible(baseline_plan, ebnf_grammar_path)
        # C) Custom Complex Structure Check
        custom_resp = check_complex_structure(target_plan)
        custom_base = check_complex_structure(baseline_plan)

        # Combine all scores into a single record
        combined_scores = {
            'case_id': case_id,
            'cg_gain': cg_scores['completeness_gain'],
            'cg_resp': cg_scores['c_resp'],
            'cg_base': cg_scores['c_base'],
            'psa_resp': psa_resp,
            'psa_base': psa_base,
            'custom_resp': float(custom_resp),  # Convert boolean to float
            'custom_base': float(custom_base)
        }
        evaluation_results.append(combined_scores)

    # --- 4. Display Final Summary ---
    print("\n" + "=" * 80)
    print("Final Combined Evaluation Summary")
    print("=" * 80)

    if not evaluation_results:
        print("No results to display.")
    else:
        results_df = pd.DataFrame(evaluation_results)
        print("Per-Case Results:")
        print(results_df.set_index('case_id').round(4))

        print("\n--- Overall Averages ---")
        avg_scores = results_df.drop(columns=['case_id']).mean()
        print(avg_scores.round(4))