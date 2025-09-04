# scripts/06_run_hybrid_experiment.py
"""
Orchestrator for running the dissertation experiment with the final
Query-Aware Hybrid Retriever.

This script is driven by a YAML configuration file. It iterates through
all specified grammar types and corpus sizes, initialises the hybrid RAG
pipeline, runs a standard set of queries, and saves the collated
results to a single JSONL file.

Usage:
    python scripts/06_run_hybrid_experiment.py --config configs/experiments/dissertation_experiment_config.yaml
"""
import os
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict

# --- Environment settings for E5 model on CPU to avoid deadlocks ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Use your centralised utility functions ---
from evaluation.retrieval.utils import (
    load_knowledge_base,
    initialise_retrievers as initialise_hybrid_retrievers
)

# --- Core RAG component imports ---
from rag.pipeline import RAGPipeline
from rag.retrieval_fusion.strategies import query_aware_fusion
from rag.generation.generator import Generator
from rag.generation.prompt_constructor import DynamicPromptConstructor

# --- Load Env ---
from dotenv import load_dotenv
load_dotenv()


def load_and_format_queries(project_root: Path, query_path_templates: List[str], context: Dict) -> List[Dict]:
    """
    Loads all specified query sets, formatting paths dynamically.
    """
    print("🔎 Loading all specified query sets...")
    all_queries = []
    for path_template in query_path_templates:
        # Format the path with the current run's context (grammar_type, corpus_size, etc.)
        formatted_path_str = path_template.format(**context)
        full_path = project_root / formatted_path_str

        if not full_path.exists():
            print(f"   - ⚠️ WARNING: Query file not found at {full_path}, skipping.")
            continue
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
                all_queries.extend(queries)
                print(f"   - ✅ Loaded {len(queries)} queries from {formatted_path_str}")
        except json.JSONDecodeError:
            print(f"   - ❌ ERROR: Could not decode JSON from {full_path}. It might be empty or malformed.")

    print(f"   Total queries to run: {len(all_queries)}\n")
    return all_queries


def main(config_path: str):
    """ Main function to run the hybrid RAG experiment suite. """
    project_root = Path(__file__).resolve().parent.parent

    # 1. Load Experiment Configuration
    print("🚀 Starting Hybrid RAG Experiment Run...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Strategy used is query_aware_fusion ---
    strategy = "query_aware_fusion"
    print(f"   - Strategy: {strategy}")

    # 2. Prepare Output File
    output_dir = project_root / config['paths']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = config['experiment_name_template'].format(strategy=strategy)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_{timestamp}.jsonl"
    output_path = output_dir / output_filename

    print(f"   - Experiment Name: {base_name}")
    print(f"   - Results will be saved to: {output_path}\n")

    # 3. Initialise Generator and Prompt Constructor (shared across all runs)
    generator = Generator(model=config['llm_model'])
    prompt_constructor = DynamicPromptConstructor()

    all_results = []
    run_id_counter = 1

    # 4. --- Main Experiment Loop ---
    for grammar in config['grammar_types']:
        for size in config['corpus_sizes']:
            current_run_desc = f"Grammar: {grammar}, Size: {size}"
            print(f"--- 🏃 Running suite for: {current_run_desc} ---")

            context = {
                'grammar_type': grammar,
                'corpus_size': size,
                'grammar_type_short': grammar.replace('_grammar', '')
            }

            # Dynamic Path logic
            output_dir_template = config['paths']['output_dir']
            dynamic_output_dir_str = output_dir_template.format(**context)
            output_dir = project_root / dynamic_output_dir_str
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create a more descriptive base name for the file
            base_name_template = config['experiment_name_template'].format(strategy=strategy)
            base_name = f"{base_name_template}_{grammar}_{size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{timestamp}.jsonl"
            output_path = output_dir / output_filename

            print(f"   - Results for this run will be saved to: {output_path}")

            # A. Load the correct knowledge base
            corpus_path_str = config['paths']['corpus_template'].format(**context)
            corpus_path = project_root / corpus_path_str
            try:
                knowledge_base = load_knowledge_base(str(corpus_path))
                print(f"   - Knowledge base loaded ({len(knowledge_base)} docs).")
            except FileNotFoundError as e:
                print(f"   - ❌ ERROR: {e}. Skipping this run.")
                continue

            # B. Load the query sets for this run
            queries = load_and_format_queries(project_root, config['paths']['query_sets'], context)

            # C. Initialise all retrievers for the hybrid model using the util function
            retrievers_dict = initialise_hybrid_retrievers(grammar, knowledge_base, project_root, size)

            # D. Build the RAG pipeline with the query-aware fusion strategy
            pipeline = RAGPipeline(
                retrievers=list(retrievers_dict.values()),
                fusion_strategy=query_aware_fusion,
                generator=generator,
                prompt_constructor=prompt_constructor
            )

            # E. Run all queries through the configured pipeline
            print(f"   - Processing {len(queries)} queries for this suite...")
            for query_info in tqdm(queries, desc=current_run_desc, unit="query"):
                try:
                    result = pipeline.run(
                        query=query_info['text'],
                        top_k=config['top_k']
                    )

                    # F. Format and store the result
                    output_entry = {
                        "run_id": f"run_{run_id_counter:04d}",
                        "experiment_name": base_name,
                        "grammar_type": grammar,
                        "corpus_size": size,
                        "retrieval_strategy": strategy,
                        "query_id": query_info.get('query_id'),
                        "query_type": query_info.get('type'),
                        "question": query_info['text'],
                        "answer": result['answer'],
                        "contexts": [doc.get("contents", "") for doc in result["retrieved_docs"]],
                        "retrieved_documents_info": [
                            {
                                "id": doc.get("id") or doc.get("session_id"),
                                "scores": {k.replace('_score', ''): v for k, v in doc.items() if '_score' in k}
                            } for doc in result["retrieved_docs"]                        ],
                        "ground_truth": ""
                    }
                    all_results.append(output_entry)
                    run_id_counter += 1

                except Exception as e:
                    print(f"\n   - ❌ ERROR processing query '{query_info.get('query_id')}': {e}")

            print(f"--- ✅ Suite complete for: {current_run_desc} ---\n")

    # 5. Save All Collated Results
    print(f"💾 Saving all {len(all_results)} results to file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n--- 🎉 All experiments finished! ---")
    print(f"   Final results are located at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hybrid RAG experiment from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file."
    )
    args = parser.parse_args()
    main(args.config)