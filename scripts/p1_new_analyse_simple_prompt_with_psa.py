# scripts/p1_run_psa_evaluation.py

import yaml
import json
from pathlib import Path
import sys
import os
import pandas as pd
from dotenv import load_dotenv
from functools import partial

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# --- Core RAG & Evaluation Component Imports ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.retrieval_fusion.strategies import dynamic_query_aware_score_fusion
from rag.generation.generator import Generator
from rag.utils import load_and_format_config
from evaluation.utils.context_metrics.structure_metrics import calculate_psa_flexible

# Import the final pipeline and the new v2 prompt constructor
from rag.pipeline import RAGPipeline
from rag.generation.prompt_constructor import PromptConstructor_v2


def initialise_components(grammar_type: str, corpus_size: int) -> tuple:
    """
    Sets up and initialises all necessary RAG retrievers and the generator.
    """
    print(f"--- Initialising RAG components for [{grammar_type.upper()}] grammar (size {corpus_size}) ---")
    context = {"grammar_type": grammar_type, "corpus_size": corpus_size}
    corpus_path = PROJECT_ROOT / f"data/processed/{grammar_type}_grammar/{grammar_type}_{corpus_size}.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    raw_corpus = [json.loads(line) for line in open(corpus_path, 'r', encoding='utf-8')]

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

    retrievers = {
        "field_metadata": field_retriever,
        "sparse_bm25": sparse_retriever,
        "semantic_e5": dense_retriever
    }
    generator = Generator(model="gpt-4o")
    print("✅ All components initialised.")
    return retrievers, generator


if __name__ == "__main__":
    # --- Experiment Parameters ---
    GRAMMAR = "high_constraint"
    CORPUS_SIZE = 100
    RETRIEVAL_TOP_K = 30
    CONTEXT_TOP_K = 3
    GENERATION_TEMPERATURE = 0.0

    EBNF_GRAMMAR_PATH = PROJECT_ROOT / "grammar/sports/squash/high_constraint_grammar/strict_structures.ebnf"
    PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts/light_touch.txt"  # Path to your prompt file

    queries_to_evaluate = [
        {"query_id": "complex_45_adv_volley",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
        {"query_id": "vague_02", "text": "generate a session to improve my forehand"},
        {"query_id": "ooc_06", "text": "a solo to practice cross drops"}
    ]

    # --- SETUP & PIPELINE ASSEMBLY ---

    # 1. Initialise components
    all_retrievers_dict, llm_generator = initialise_components(GRAMMAR, CORPUS_SIZE)
    with open(PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml", "r") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    # 2. Initialise the prompt constructor with the specific template file
    prompt_constructor = PromptConstructor_v2(template_path=PROMPT_TEMPLATE_PATH)

    # 3. Prepare the fusion strategy with its required config using 'partial'
    query_aware_fusion_strategy = partial(
        dynamic_query_aware_score_fusion,
        field_scoring_config=field_scoring_config
    )

    # 4. Assemble the RAG Pipeline with all the final components
    rag_pipeline = RAGPipeline(
        retrievers=list(all_retrievers_dict.values()),
        fusion_strategy=query_aware_fusion_strategy,
        prompt_constructor=prompt_constructor,
        generator=llm_generator
    )

    # --- MAIN EVALUATION LOOP ---
    query_results = []
    for query_info in queries_to_evaluate:
        query_id, query_text = query_info["query_id"], query_info["text"]
        print("\n" + "=" * 80)
        print(f"Processing query: {query_id}")

        # --- STAGE 1: GENERATE PLAN (using the pipeline) ---
        print(f"  -> Generating plan with RAG pipeline (retrieval_k={RETRIEVAL_TOP_K}, context_k={CONTEXT_TOP_K})...")
        pipeline_result = rag_pipeline.run(
            query=query_text,
            retrieval_k=RETRIEVAL_TOP_K,
            context_k=CONTEXT_TOP_K
        )
        generated_plan = pipeline_result["answer"]

        # --- STAGE 2: EVALUATION with PSA Flexible ---
        print("  -> Evaluating with Flexible PSA...")
        psa_score = calculate_psa_flexible(generated_plan, EBNF_GRAMMAR_PATH)
        result = {'query_id': query_id, 'psa_score': psa_score}
        print(f"  -> PSA Score: {result['psa_score']:.4f}")
        query_results.append(result)

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("Final PSA Evaluation Summary (Simple Prompt)")
    print("=" * 80)
    results_df = pd.DataFrame(query_results)
    print("Per-Query Results:")
    print(results_df.set_index('query_id').round(4))
    print("\nOverall Average PSA:")
    avg_psa = results_df['psa_score'].mean()
    print(f"{avg_psa:.4f}")