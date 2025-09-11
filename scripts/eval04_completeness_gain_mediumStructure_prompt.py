# scripts/eval_completeness_gain.py

import yaml
import json
from pathlib import Path
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Add other environment variables as per project standards
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# --- Core RAG & Evaluation Component Imports ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.retrieval_fusion.strategies import dynamic_query_aware_rrf
from rag.generation.generator import Generator
from rag.utils import load_and_format_config
from evaluation.utils.completeness_gain import CompletenessGain


def initialise_components(grammar_type: str, corpus_size: int) -> tuple:
    """
    A helper function to set up and initialise all necessary RAG components.
    This function is replicated from other evaluation scripts for consistency.
    """
    print(f"--- Initialising RAG components for [{grammar_type.upper()}] grammar (size {corpus_size}) ---")
    context = {"grammar_type": grammar_type, "corpus_size": corpus_size}
    corpus_path = PROJECT_ROOT / f"data/processed/{grammar_type}_grammar/{grammar_type}_{corpus_size}.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    raw_corpus = [json.loads(line) for line in open(corpus_path, 'r', encoding='utf-8')]

    # Initialise retrievers... (logic copied from existing scripts)
    adapter = SquashNewCorpusAdapter()
    adapted_corpus = [adapter.transform(doc) for doc in raw_corpus]
    field_retriever = FieldRetriever(knowledge_base=adapted_corpus,
                                     config_path=PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml")

    sparse_config = load_and_format_config(str(PROJECT_ROOT / "configs/retrieval/sparse_retriever.yaml"), context)
    sparse_config['sparse_params']['index_path'] = str(PROJECT_ROOT / sparse_config['sparse_params']['index_path'])
    sparse_retriever = SparseRetriever(knowledge_base=raw_corpus, config=sparse_config['sparse_params'])

    dense_config = load_and_format_config(str(PROJECT_ROOT / "configs/retrieval/semantic_retriever.yaml"), context)
    dense_config['index_path'] = str(PROJECT_ROOT / dense_config['index_path'])
    dense_retriever = SemanticRetriever(config=dense_config)

    retrievers = {"field": field_retriever, "sparse": sparse_retriever, "dense": dense_retriever}

    # Initialise generator
    generator = Generator(model="gpt-4o")
    print("✅ All components initialised.")
    return retrievers, generator


if __name__ == "__main__":
    # --- Experiment Parameters ---
    GRAMMAR = "high_constraint"
    CORPUS_SIZE = 100
    RETRIEVAL_TOP_K = 30
    CONTEXT_TOP_K = 3
    # Use zero temperature for deterministic output to isolate the impact of context.
    GENERATION_TEMPERATURE = 0.0
    # Path to the grammar file, used by the evaluator to define structural "atoms".
    GRAMMAR_PATHS = {
        "balanced": "grammar/sports/squash/balanced_grammar/loose_structures.ebnf",
        "high_constraint": "grammar/sports/squash/high_constraint_grammar/strict_structures.ebnf",
        "loose": "grammar/sports/squash/loose_grammar/loose_structures.ebnf"
    }

    # Select the correct path based on the GRAMMAR variable and handle errors.
    if GRAMMAR not in GRAMMAR_PATHS:
        print(f"Error: Invalid GRAMMAR name '{GRAMMAR}'. Please choose from {list(GRAMMAR_PATHS.keys())}.")
        sys.exit(1)
    EBNF_GRAMMAR_PATH = GRAMMAR_PATHS[GRAMMAR]

    queries_to_evaluate = [
        {"query_id": "complex_45_adv_volley",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
        {"query_id": "vague_02", "text": "generate a session to improve my forehand"},
        {"query_id": "ooc_06", "text": "a solo to practice cross drops"}
    ]

    # --- SETUP ---
    all_retrievers, llm_generator = initialise_components(GRAMMAR, CORPUS_SIZE)
    with open(PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml", "r") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    print(f"\n--- Initialising CompletenessGain evaluator with grammar: {EBNF_GRAMMAR_PATH} ---")
    completeness_evaluator = CompletenessGain(ebnf_grammar_path=PROJECT_ROOT / EBNF_GRAMMAR_PATH)

    query_results = []

    # --- Main Evaluation Loop ---
    for query_info in queries_to_evaluate:
        query_id, query_text = query_info["query_id"], query_info["text"]
        print("\n" + "=" * 80)
        print(f"Processing query: {query_id}")

        # --- STAGE 1: GENERATE TARGET PLAN (with RAG) ---
        print("  -> Generating TARGET plan (with RAG context)...")
        standalone_results = {n: r.search(query=query_text, top_k=RETRIEVAL_TOP_K) for n, r in all_retrievers.items()}
        fused_documents = dynamic_query_aware_rrf(standalone_results, query_text, field_scoring_config)
        context_docs = fused_documents[:CONTEXT_TOP_K]
        rag_context_str = "\n\n---\n\n".join([doc['contents'] for doc in context_docs])

        prompt_template = """You are an expert squash coach AI that generates session plans.

        Your task is to generate a detailed squash session plan based on the information provided in the "CONTEXT" section.

        The final output must be a single, complete, and well-structured session plan. 
        You should structure the plan using markdown headers for key sections like "Warm-up" and "Activity".

        CONTEXT:
        {context}

        USER REQUEST: "{query}"

        FINAL SQUASH SESSION PLAN:
        """

        target_prompt = prompt_template.format(query=query_text, context=rag_context_str)
        target_plan = llm_generator.generate(target_prompt, temperature=GENERATION_TEMPERATURE)

        # --- STAGE 2: GENERATE BASELINE PLAN (without RAG) ---
        print("  -> Generating BASELINE plan (without RAG context)...")
        baseline_context = "No external context provided. Generate a plan based on general knowledge."
        baseline_prompt = prompt_template.format(query=query_text, context=baseline_context)
        baseline_plan = llm_generator.generate(baseline_prompt, temperature=GENERATION_TEMPERATURE)

        # --- STAGE 3: EVALUATION with CompletenessGain ---
        print("  -> Evaluating with CompletenessGain...")
        scores = completeness_evaluator.get_score(generated_plan=target_plan, baseline_text=baseline_plan)
        scores['query_id'] = query_id

        print(f"  -> Completeness Scores: {scores}")
        query_results.append(scores)

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("Final Completeness Gain Evaluation Summary")
    print("=" * 80)
    results_df = pd.DataFrame(query_results)

    print("Per-Query Results:")
    print(results_df.set_index('query_id').round(4))

    print("\nOverall Averages:")
    avg_scores = results_df[['completeness_gain', 'c_resp', 'c_base', 'k_size']].mean()
    print(avg_scores.round(4))