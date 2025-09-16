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

from rag.parsers.user_query_parser import parse_type
from evaluation.utils.context_metrics.structure_metrics import calculate_pda, calculate_stc


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
    GRAMMAR = "high_constraint"  # Can be "high_constraint", "balanced", or "loose"
    CORPUS_SIZE = 100
    RETRIEVAL_TOP_K = 30
    CONTEXT_TOP_K = 3
    GENERATION_TEMPERATURE = 0.0

    # --- UPDATED: Configuration for Metrics ---
    # Path to the config file containing the point-to-duration conversion factor for PDA.
    # NOTE: This file should contain the `point_duration_minutes` value.
    SESSION_TYPES_CONFIG_PATH = PROJECT_ROOT / "configs/session_types.yaml"

    # Dynamic path mapping for structure templates based on your project's structure.
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
            # Loose grammar only has one path; other types will default to it.
            "mix": "grammar/sports/squash/loose_grammar/session_structures/mix/"
        }
    }

    queries_to_evaluate = [
        {"query_id": "complex_01_cg", "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_02_cg", "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_03_cg",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
        {"query_id": "complex_21_mix", "text": "a 60-minute mix session"},
        {"query_id": "complex_22_mix", "text": "a 60-minute mix session for an intermediate player"},
        {"query_id": "complex_23_mix",
         "text": "a 60-minute mix session for an intermediate player focusing on straight lob"},
        {"query_id": "complex_24_mix",
         "text": "a 60-minute mix session for an intermediate player focusing on forehand straight kill"},
        {"query_id": "ooc_01", "text": "a session focusing on the volley cross"},
        {"query_id": "ooc_02", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_03", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_05", "text": "practice the 3-step ghosting"},
        {"query_id": "ooc_06", "text": "a solo to practice cross drops"}
    ]

    # --- SETUP ---
    all_retrievers, llm_generator = initialise_components(GRAMMAR, CORPUS_SIZE)
    with open(PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml", "r") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    query_results = []

    # --- Main Evaluation Loop ---
    for query_info in queries_to_evaluate:
        query_id, query_text = query_info["query_id"], query_info["text"]
        print("\n" + "=" * 80)
        print(f"Processing query: '{query_text}'")

        # --- STAGE 1: Determine Structure Template Path from Query ---
        print("  -> Parsing query to determine structure template...")
        session_type = parse_type(query_text)

        # Look up the directory path based on grammar and session type
        grammar_paths = STRUCTURE_DIRECTORIES.get(GRAMMAR, {})
        template_dir_path_str = grammar_paths.get(session_type)

        # Fallback for loose grammar or vague queries
        if not template_dir_path_str:
            session_type = "mix"  # Default to 'mix'
            template_dir_path_str = grammar_paths.get(session_type)

        structure_template_path = None
        if template_dir_path_str:
            template_dir = PROJECT_ROOT / template_dir_path_str
            if template_dir.is_dir():
                # Assumption: For this evaluation, we pick the first available YAML file.
                yaml_files = sorted(list(template_dir.glob("*.yaml")))
                if yaml_files:
                    structure_template_path = yaml_files[0]
                    print(f"  -> Selected template for PDA: {structure_template_path.name}")

        # --- STAGE 2: Retrieval & Generation ---
        standalone_results = {n: r.search(query=query_text, top_k=RETRIEVAL_TOP_K) for n, r in all_retrievers.items()}
        fused_documents = dynamic_query_aware_rrf(standalone_results, query_text, field_scoring_config)
        context_docs = fused_documents[:CONTEXT_TOP_K]
        rag_context_str = "\n\n---\n\n".join([doc['contents'] for doc in context_docs])

        prompt_template = "You are an expert squash coach AI... (Your preferred prompt here)"  # Add your prompt
        final_prompt = prompt_template.format(query=query_text, context=rag_context_str)
        generated_plan = llm_generator.generate(final_prompt, temperature=GENERATION_TEMPERATURE)

        # --- STAGE 3: Evaluation with PDA and STC ---
        print("  -> Evaluating with PDA and STC...")

        pda_score = 0.0
        if structure_template_path:
            pda_score = calculate_pda(
                generated_plan_text=generated_plan,
                source_structure_template_path=structure_template_path,
                session_types_config_path=SESSION_TYPES_CONFIG_PATH
            )
        else:
            print("  -> WARNING: Could not find a suitable structure template. PDA score defaulted to 0.0.")

        stc_score = calculate_stc(
            generated_plan_text=generated_plan,
            user_query=query_text
        )

        scores = {'query_id': query_id, 'pda_score': pda_score, 'stc_score': stc_score}
        print(f"  -> Scores: {scores}")
        query_results.append(scores)

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("Final PDA & STC Evaluation Summary")
    print("=" * 80)
    results_df = pd.DataFrame(query_results)

    print("Per-Query Results:")
    print(results_df.set_index('query_id').round(4))

    print("\nOverall Averages:")
    avg_scores = results_df.drop(columns=['query_id']).mean()
    print(avg_scores.round(4))