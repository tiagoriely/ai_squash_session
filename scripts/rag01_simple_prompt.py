# scripts/run_single_test.py

# Indivual retrievers top_k = 30
# Query_aware_fusion top_k = 3
#

import sys
from pathlib import Path
import os
import json
import yaml

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Path and Environment Setup ---
# This ensures the script can find your custom modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()  # Load environment variables from .env file

# --- Core RAG Component Imports ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.retrieval_fusion.strategies import dynamic_query_aware_rrf
from rag.generation.generator import Generator
from rag.utils import load_and_format_config


def initialise_components(grammar_type: str, corpus_size: int) -> tuple:
    """A helper function to set up and initialise all necessary RAG components."""
    print(f"--- Initialising RAG components for [{grammar_type.upper()}] grammar (size {corpus_size}) ---")

    context = {"grammar_type": grammar_type, "corpus_size": corpus_size}

    # --- 1. Load Corpus (needed by multiple retrievers) ---
    corpus_path_str = f"data/processed/{grammar_type}_grammar/{grammar_type}_{corpus_size}.jsonl"
    corpus_path = PROJECT_ROOT / corpus_path_str
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    raw_corpus = [json.loads(line) for line in open(corpus_path, 'r', encoding='utf-8')]

    # --- 2. Initialise Retrievers ---
    # Field Retriever
    adapter = SquashNewCorpusAdapter()
    adapted_corpus = [adapter.transform(doc) for doc in raw_corpus]
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    field_retriever = FieldRetriever(knowledge_base=adapted_corpus, config_path=field_config_path)

    # Metadata Sparse Retriever
    sparse_config_path = PROJECT_ROOT / "configs/retrieval/sparse_retriever.yaml"
    sparse_config = load_and_format_config(str(sparse_config_path), context)
    sparse_config['sparse_params']['index_path'] = str(PROJECT_ROOT / sparse_config['sparse_params']['index_path'])
    sparse_retriever = SparseRetriever(knowledge_base=raw_corpus, config=sparse_config['sparse_params'])

    # Dense Retriever
    dense_config_path = PROJECT_ROOT / "configs/retrieval/semantic_retriever.yaml"
    dense_config = load_and_format_config(str(dense_config_path), context)
    dense_config['corpus_path'] = str(PROJECT_ROOT / dense_config['corpus_path'])
    dense_config['index_path'] = str(PROJECT_ROOT / dense_config['index_path'])
    dense_retriever = SemanticRetriever(config=dense_config)

    retrievers = {
        "field_metadata": field_retriever,
        "sparse_bm25": sparse_retriever,
        "semantic_e5": dense_retriever
    }

    # --- 3. Initialise Generator ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    generator = Generator(model="gpt-4o")

    print("✅ All components initialised.")
    return retrievers, generator


if __name__ == "__main__":
    # --- Experiment Parameters ---
    GRAMMAR = "balanced"
    CORPUS_SIZE = 100
    QUERY = "a 45-minute conditioned game session for an advanced player focusing on volley drops"
    TOP_K_CONTEXT = 1  # Number of retrieved documents to use as context

    queries_to_run = [
        {
            "query_id": "human_test_01",
            "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"
        },
        {
            "query_id": "human_test_02",
            "text": "a 60-minute mix session for an intermediate player focusing on the forehand straight kill"
        },
        {
            "query_id": "human_test_03",
            "text": "a session to improve my movement to the front"
        }
    ]

    # ----------------------------------------------------------------------
    #  RAG PIPELINE EXECUTION
    # ----------------------------------------------------------------------

    # --- SETUP ---
    # Load the Field Retriever's scoring config for the dynamic fusion strategy
    field_cfg_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    with open(field_cfg_path, "r", encoding="utf-8") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    # Initialise all retrievers and the generator
    all_retrievers, llm_generator = initialise_components(GRAMMAR, CORPUS_SIZE)

    all_generated_results = []
    for query_info in queries_to_run:
        query_id = query_info["query_id"]
        query_text = query_info["text"]

        print("\n" + "=" * 80)
        print(f"Processing query_id: {query_id}")

    # --- STAGE 1: RETRIEVAL & FUSION ---
    print("\n--- [Stage 1: Retrieval & Fusion] ---")
    standalone_results = {
        name: retriever.search(query=QUERY, top_k=30)
        for name, retriever in all_retrievers.items()
    }

    # Apply the dynamic hybrid fusion strategy
    fused_documents = dynamic_query_aware_rrf(
        ranked_lists_map=standalone_results,
        query=QUERY,
        field_scoring_config=field_scoring_config
    )

    # --- STAGE 2: CONTEXT FORMULATION ---
    print("\n--- [Stage 2: Context Formulation] ---")
    context_docs = fused_documents[:TOP_K_CONTEXT]
    context_str = "\n\n---\n\n".join(
        [f"Source Document ID: {doc.get('id', 'N/A')}\n\n{doc['contents']}" for doc in context_docs])

    print(f"Using top {len(context_docs)} documents as context for the generator.")
    print(f"Top retrieved document IDs: {[doc.get('id') for doc in context_docs]}")

    # For your analysis, you can inspect the full context
    # print("\n--- RETRIEVED CONTEXT ---\n")
    # print(context_str)

    # --- STAGE 3: GENERATION ---
    print("\n--- [Stage 3: Generation] ---")

    prompt_template = """
You are an expert squash coach AI. Your task is to generate a detailed and coherent squash session plan.

Generate the plan based *only* on the information provided in the "CONTEXT" section below. Do not add any exercises or information not present in the context.

The user's request is: "{query}"

Strictly adhere to all constraints mentioned in the user's request, such as duration, player level, and shot focus. The final output must be a single, complete, and well-structured session plan.

CONTEXT:
{context}

FINAL SQUASH SESSION PLAN:
"""

    final_prompt = prompt_template.format(query=QUERY, context=context_str)

    print("Sending final prompt to the generator...")
    generated_plan = llm_generator.generate(final_prompt)

    # Store the result in the required format ---
    all_generated_results.append({
        "query_id": query_id,
        "query_text": query_text,
        "generated_plan": generated_plan
    })
    print(f"  -> Plan generated for {query_id}.")

    # Save all results to a single JSON file ---
output_path = PROJECT_ROOT / "generated_plans.json"
print("\n" + "=" * 80)
print(f"Saving all generated plans to: {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_generated_results, f, indent=2)
print("✅ JSON file saved successfully.")