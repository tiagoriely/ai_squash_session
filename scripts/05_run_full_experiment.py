# scripts/05_run_full_experiment.py
"""
Orchestrator for running a full suite of RAG experiments for the dissertation.

This script is driven by a single YAML configuration file. It iterates through
all specified grammar types and corpus sizes, initialises the appropriate RAG
pipeline for each, runs a standard set of queries, and saves the collated
results to a single JSONL file.

Usage:
    python scripts/05_run_full_experiment.py --config configs/experiments/dissertation_experiment_config.yaml
"""
import os
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# --- Environment settings for E5 model on CPU ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Core RAG component imports ---
from rag.pipeline import RAGPipeline
from rag.retrieval.semantic_retriever import SemanticRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval_fusion import reciprocal_rank_fusion
from rag.generation.generator import Generator
from rag.generation.prompt_constructor import DynamicPromptConstructor
from rag.utils import load_and_format_config
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter

# --- Load Env ---
from dotenv import load_dotenv
load_dotenv()


# --- Helper Functions ---

def load_queries(project_root: Path, query_paths: List[str]) -> List[Dict]:
    """Loads all specified query sets into a single list."""
    print("🔎 Loading all specified query sets...")
    all_queries = []
    for path_str in query_paths:
        full_path = project_root / path_str
        if not full_path.exists():
            print(f"   - ⚠️ WARNING: Query file not found at {full_path}, skipping.")
            continue
        with open(full_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
            all_queries.extend(queries)
            print(f"   - ✅ Loaded {len(queries)} queries from {path_str}")
    print(f"   Total queries to run: {len(all_queries)}\n")
    return all_queries


def load_knowledge_base(corpus_path: Path) -> List[Dict]:
    """Loads the corpus from a JSONL file."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def initialise_retrievers(
        config: Dict,
        context: Dict,
        knowledge_base: List[Dict],
        project_root: Path
) -> List[Any]:
    """
    Initialises and returns a list of retriever instances based on the
    experiment's retrieval strategy.
    """
    strategy = config['retrieval_strategy']
    retrievers = []

    # This context is for the component configs (semantic.yaml, sparse.yaml)
    # which expect the short grammar name (e.g., 'balanced') for their templates.
    component_context = {
        'grammar_type': context['grammar_type_short'],
        'corpus_size': context['corpus_size']
    }

    print(f"   - Initialising retrievers for strategy: '{strategy}'")

    if strategy in ['semantic_only', 'hybrid_rrf']:
        semantic_config_path = project_root / config['paths']['retriever_configs']['semantic']
        semantic_config = load_and_format_config(str(semantic_config_path), component_context)
        # Ensure paths within the loaded config are absolute
        semantic_config['index_path'] = str(project_root / semantic_config['index_path'])
        semantic_config['corpus_path'] = str(project_root / semantic_config['corpus_path'])
        retrievers.append(SemanticRetriever(config=semantic_config))
        print("     - SemanticRetriever (e5) initialised.")

    if strategy in ['sparse_only', 'hybrid_rrf']:
        sparse_config_path = project_root / config['paths']['retriever_configs']['sparse']
        sparse_config = load_and_format_config(str(sparse_config_path), component_context)
        retrievers.append(SparseRetriever(knowledge_base=knowledge_base, config=sparse_config['sparse_params']))
        print("     - SparseRetriever (BM25) initialised.")

    if strategy in ['field_only', 'hybrid_rrf']:
        adapter = SquashNewCorpusAdapter()
        canonical_kb = [adapter.transform(doc) for doc in knowledge_base]
        field_config_path = project_root / config['paths']['retriever_configs']['field']
        retrievers.append(FieldRetriever(knowledge_base=canonical_kb, config_path=str(field_config_path)))
        print("     - FieldRetriever (Metadata) initialised.")

    if not retrievers:
        raise ValueError(f"Unknown or unsupported retrieval strategy: {strategy}")

    return retrievers


# --- Main Orchestration Logic ---

def main(config_path: str):
    """
    Main function to run the entire experiment suite.
    """
    project_root = Path(__file__).resolve().parent.parent

    # 1. Load Experiment Configuration
    print(f"🚀 Starting Experiment Run...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    strategy = config['retrieval_strategy']
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

    # 3. Load All Queries
    queries = load_queries(project_root, config['paths']['query_sets'])

    # 4. Initialise Generator and Prompt Constructor (shared across all runs)
    generator = Generator(model=config['llm_model'])
    prompt_constructor = DynamicPromptConstructor()

    all_results = []
    run_id_counter = 1

    # 5. --- Main Experiment Loop ---
    for grammar in config['grammar_types']:
        for size in config['corpus_sizes']:

            current_run_desc = f"Grammar: {grammar}, Size: {size}"
            print(f"--- 🏃 Running suite for: {current_run_desc} ---")

            # A. Create dynamic context for this specific run
            grammar_short_name = grammar.replace('_grammar', '')
            context = {
                'grammar_type': grammar,
                'corpus_size': size,
                'grammar_type_short': grammar_short_name
            }

            # B. Load the correct knowledge base
            corpus_path_str = config['paths']['corpus_template'].format(**context)
            corpus_path = project_root / corpus_path_str
            try:
                knowledge_base = load_knowledge_base(corpus_path)
                print(f"   - Knowledge base loaded ({len(knowledge_base)} docs).")
            except FileNotFoundError as e:
                print(f"   - ❌ ERROR: {e}. Skipping this run.")
                continue

            # C. Initialise the retriever(s) for this run
            retrievers = initialise_retrievers(config, context, knowledge_base, project_root)

            # D. Configure the fusion strategy
            fusion_strategy = reciprocal_rank_fusion if config['retrieval_strategy'] == 'hybrid_rrf' else None

            # E. Build the RAG pipeline for this configuration
            pipeline = RAGPipeline(
                retrievers=retrievers,
                fusion_strategy=fusion_strategy,
                generator=generator,
                prompt_constructor=prompt_constructor
            )

            # F. Run all queries through the configured pipeline
            print(f"   - Processing {len(queries)} queries...")
            for query_info in tqdm(queries, desc=current_run_desc, unit="query"):
                try:
                    run_params = {
                        'query': query_info['text'],
                        'top_k': config['top_k']
                    }

                    if config['retrieval_strategy'] == 'hybrid_rrf':
                        run_params['fusion_k'] = config.get('rrf_k')

                    result = pipeline.run(**run_params)

                    # G. Format and store the result
                    output_entry = {
                        "run_id": f"run_{run_id_counter:04d}",
                        "experiment_name": base_name,
                        "grammar_type": grammar,
                        "corpus_size": size,
                        "retrieval_strategy": config['retrieval_strategy'],
                        "query_id": query_info.get('query_id'),
                        "query_type": query_info.get('type'),
                        "question": query_info['text'],
                        "answer": result['answer'],
                        "contexts": [doc.get("contents", "") for doc in result["retrieved_docs"]],
                        "retrieved_documents_info": [
                            {"id": doc.get("id"), "score": doc.get("score", {})} for doc in result["retrieved_docs"]
                        ],
                        "ground_truth": ""  # Placeholder for Ragas evaluation
                    }
                    all_results.append(output_entry)
                    run_id_counter += 1

                except Exception as e:
                    print(f"\n   - ❌ ERROR processing query '{query_info.get('query_id')}': {e}")

            print(f"--- ✅ Suite complete for: {current_run_desc} ---\n")

    # 6. Save All Collated Results
    print(f"💾 Saving all {len(all_results)} results to file...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n--- 🎉 All experiments finished! ---")
    print(f"   Final results are located at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full suite of RAG experiments from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file."
    )
    args = parser.parse_args()
    main(args.config)