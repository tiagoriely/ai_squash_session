# In evaluation/evaluate_retrievers.py
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from pathlib import Path  # Import the Path object

# --- Core Imports from your RAG library ---
from rag.retrieval.semantic_retriever import SemanticRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.field_retriever import FieldRetriever
from rag.utils import load_and_format_config

from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter



# HEEEEEELL YEEEEAH
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# --- Constants ---
QUERIES_FILE_PATH = 'evaluation/test_queries.json'
TOP_K = 10


# --- Helper Functions ---
def load_knowledge_base(corpus_path: str) -> list[dict]:
    """Loads the corpus from a JSONL file."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    print(f"Loading knowledge base from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_test_queries(filepath: str):
    """Loads queries from the specified JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_query_sets(project_root: Path, grammar_type: str, corpus_size: int) -> list[dict]:
    """Finds and loads all generated query sets into a single list."""

    print("Loading all query sets...")
    all_queries = []
    query_set_dir = project_root / "evaluation" / "query_sets" / "generated"

    # 1. Load the static query sets
    static_sets = [
        "out_of_distribution.json",
        "under_specified.json",
        "graduated_complexity.json"
    ]
    for filename in static_sets:
        path = query_set_dir / filename
        with open(path, 'r') as f:
            all_queries.extend(json.load(f))
            print(f"   - Loaded {filename}")

    # 2. Load the dynamic "golden set" for the specific grammar
    golden_set_path = query_set_dir / grammar_type / str(corpus_size) / "golden_set.json"
    if golden_set_path.exists():
        with open(golden_set_path, 'r') as f:
            all_queries.extend(json.load(f))
            print(f"   - Loaded {golden_set_path.name} for {grammar_type}")
    else:
        print(f"   - WARNING: Golden set not found at {golden_set_path}")

    print(f"Total queries loaded: {len(all_queries)}")
    return all_queries

def run_evaluation(grammar_type: str):
    """Main function to run the evaluation pipeline for a given grammar."""

    # 1. Get the directory of the current script (which is inside evaluation/)
    script_dir = Path(__file__).resolve().parent.parent
    # 2. Get the project root directory by going up one level
    project_root = script_dir.parent

    # --- Use the absolute path to find the config file ---
    # Load a retriever config to get the corpus_size, instead of hardcoding it.
    semantic_config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    # We load it without context first to read the variable
    base_semantic_config = load_and_format_config(str(semantic_config_path))
    corpus_size = base_semantic_config['corpus_size']

    # Construct absolute path for the corpus from the project root
    # Now create the context with the dynamic corpus_size
    template_context = {'grammar_type': grammar_type.replace('_grammar', ''), 'corpus_size': corpus_size}

    # Format the config again with the context to resolve paths
    temp_config = load_and_format_config(str(semantic_config_path), template_context)
    corpus_path = project_root / temp_config['corpus_path']

    knowledge_base = load_knowledge_base(str(corpus_path))
    retrievers = initialise_retrievers(grammar_type, knowledge_base, project_root, corpus_size)

    # Pass the dynamic corpus_size to the query loader
    queries = load_all_query_sets(project_root, grammar_type, corpus_size)
    all_results = []

    print(f"\nRunning evaluation for {len(queries)} queries across {len(retrievers)} retrievers...")
    for query_info in tqdm(queries, desc="Processing Queries"):
        query_text = query_info['text']

        for retriever_name, retriever_instance in retrievers.items():
            try:
                print(f"\n--> Starting search with '{retriever_name}' for query: '{query_text[:30]}...'")

                retrieved_results = retriever_instance.search(query=query_text, top_k=TOP_K)

                print(f"<-- Finished search with '{retriever_name}'.")

                for rank, result_dict in enumerate(retrieved_results):
                    score_keys = ['semantic_score', 'sparse_score', 'field_score']
                    score = next((result_dict[key] for key in score_keys if key in result_dict), 0.0)

                    all_results.append({
                        'query_id': query_info['query_id'],
                        'query_type': query_info['type'],
                        'query_text': query_text,
                        'retriever_name': f"{retriever_name}_{grammar_type}",
                        'rank': rank + 1,
                        'document_id': result_dict.get('id') or result_dict.get('session_id'),
                        'score': score
                    })
            except Exception as e:
                print(f"\nERROR processing query '{query_info['query_id']}' with retriever '{retriever_name}': {e}")

    results_df = pd.DataFrame(all_results)

    output_dir = project_root / "evaluation" / "retrieval" / grammar_type / f"corpus_size_{corpus_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"retrieval_results_all-sets_{grammar_type}_{corpus_size}.csv"

    print(f"\n4. Saving all results to {output_filename}...")
    results_df.to_csv(output_filename, index=False)
    print("\n✅ Evaluation complete!")


# --- Initialisation ---
def initialise_retrievers(grammar_type: str, knowledge_base: list[dict], project_root: Path, corpus_size: int):
    """Initialises all retrievers for a specific grammar type using your actual config files."""
    print(f"Initialising retrievers for grammar: {grammar_type}...")

    template_context = {
        'grammar_type': grammar_type.replace('_grammar', ''),
        'corpus_size': corpus_size
    }

    # --- Use absolute paths for configs ---
    semantic_config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    sparse_config_path = project_root / "configs" / "retrieval" / "sparse_retriever.yaml"
    field_config_path = project_root / "configs" / "retrieval" / "raw_squash_field_retrieval_config.yaml"

    # --- Instantiate Retrievers ---
    semantic_config_raw = load_and_format_config(str(semantic_config_path), template_context)
    semantic_config_raw['corpus_path'] = str(project_root / semantic_config_raw['corpus_path'])

    semantic_retriever = SemanticRetriever(config=semantic_config_raw)
    sparse_config_raw = load_and_format_config(str(sparse_config_path), template_context)
    sparse_retriever = SparseRetriever(
        knowledge_base=knowledge_base,
        config=sparse_config_raw['sparse_params']
    )

    print("   - Creating canonical knowledge base for FieldRetriever...")
    adapter = SquashNewCorpusAdapter()
    canonical_kb = [adapter.transform(doc) for doc in knowledge_base]

    field_retriever = FieldRetriever(
        knowledge_base=canonical_kb,
        config_path=str(field_config_path)
    )


    retrievers = {
            'semantic_e5': semantic_retriever,
            'sparse_bm25': sparse_retriever,
            'field_metadata': field_retriever
        }

    return retrievers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retriever evaluation for a specific grammar.")
    parser.add_argument(
        "grammar",
        type=str,
        choices=['balanced_grammar', 'high_constraint_grammar', 'loose_grammar'],
        help="The type of grammar to evaluate."
    )
    args = parser.parse_args()

    run_evaluation(grammar_type=args.grammar)