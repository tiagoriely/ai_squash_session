# scripts/gen02_generate_outputs.py
import yaml
import json
from pathlib import Path
import sys
import os
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# --- Single-Corpus Path (no grammar/size switching) ---
# Point directly to your manual corpus
CORPUS_PATH = Path("/Users/tiago/projects/ai_squash_session/data/processed/my_kb_grammar/my_kb_0.jsonl")

# --- Core RAG Component Imports ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
# Adapter not needed for this manual corpus
from rag.retrieval_fusion.strategies import dynamic_query_aware_score_fusion
from rag.generation.generator import Generator
from rag.utils import load_and_format_config


def initialise_components(grammar_type: str, corpus_size: int) -> tuple:
    """A helper function to set up and initialise all necessary RAG components."""
    print(f"--- Initialising RAG components for [{grammar_type.upper()}] grammar (size {corpus_size}) ---")

    # Context still passed through (logic unchanged), even though we now use a single corpus
    context = {"grammar_type": grammar_type, "corpus_size": corpus_size}

    # --- 1. Load Corpus (single, manual) ---
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found at: {CORPUS_PATH}")
    raw_corpus = [json.loads(line) for line in open(CORPUS_PATH, 'r', encoding='utf-8')]

    # --- 2. Initialise Retrievers ---
    # Field Retriever (use raw corpus directly; adapter not required)
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    field_retriever = FieldRetriever(knowledge_base=raw_corpus, config_path=field_config_path)

    # Metadata Sparse Retriever
    sparse_config_path = PROJECT_ROOT / "configs/retrieval/sparse_retriever.yaml"
    sparse_config = load_and_format_config(str(sparse_config_path), context)
    sparse_config['sparse_params']['index_path'] = str(PROJECT_ROOT / sparse_config['sparse_params']['index_path'])
    sparse_retriever = SparseRetriever(knowledge_base=raw_corpus, config=sparse_config['sparse_params'])

    # Dense Retriever
    dense_config_path = PROJECT_ROOT / "configs/retrieval/semantic_retriever.yaml"
    dense_config = load_and_format_config(str(dense_config_path), context)
    # Point dense retriever directly at the single corpus
    dense_config['corpus_path'] = str(CORPUS_PATH)
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
    # Size is irrelevant now; keep variables to preserve logic
    CORPUS_SIZE = 0
    TOP_K_CONTEXT = 10  # This is 'k', which we will vary in later experiments
    LLM_TEMPERATURE = 0.2
    # Single grammar label to preserve outer-loop structure
    GRAMMARS_TO_TEST = ['my_kb']

    # A curated set of 5 queries for qualitative human evaluation
    queries_for_evaluation = [
        {"query_id": "complex_01_cg", "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_02_cg", "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_03_cg",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
        {"query_id": "complex_04_cg",
         "text": "a 45-minute conditioned game session for an intermediate player focusing on backhand counter drops"},
        {"query_id": "complex_21_mix", "text": "a 60-minute mix session"},
        {"query_id": "complex_22_mix", "text": "a 60-minute mix session for an intermediate player"},
        {"query_id": "complex_23_mix",
         "text": "a 60-minute mix session for an professional player focusing on straight lob"},
        {"query_id": "complex_24_mix",
         "text": "a 60-minute mix session for an intermediate player focusing on forehand straight kill"},
        {"query_id": "complex_11_drill", "text": "a 60-minute drill session"},
        {"query_id": "complex_12_drill", "text": "a 45-minute drill session focusing on the cross kill"},
        {"query_id": "complex_22_drill", "text": "a drill session to improve on the forehand 2-wall boast"},
        {"query_id": "complex_33_drill", "text": "a drill session for a professional to improve on the backhand 3-wall boast"},
    ]

    # --- SETUP ---
    field_cfg_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    with open(field_cfg_path, "r", encoding="utf-8") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    all_generated_results = []

    session_counter = 1

    # --- Main Experiment Loop ---
    # Outer loop iterates through each grammar type (now only one)
    for grammar in GRAMMARS_TO_TEST:
        try:
            all_retrievers, llm_generator = initialise_components(grammar, CORPUS_SIZE)
        except FileNotFoundError as e:
            print(f"\nSKIPPING grammar '{grammar}': {e}")
            continue

        # Inner loop iterates through the curated queries
        for query_info in tqdm(queries_for_evaluation, desc=f"Generating for [{grammar.upper()}]"):
            query_id = query_info["query_id"]
            query_text = query_info["text"]

            # --- RAG Pipeline Execution ---
            # 1. RETRIEVAL & FUSION using your dynamic hybrid retriever
            standalone_results = {name: retriever.search(query=query_text, top_k=30) for name, retriever in
                                  all_retrievers.items()}
            fused_documents = dynamic_query_aware_score_fusion(standalone_results, query_text, field_scoring_config)

            # 2. CONTEXT FORMULATION
            context_docs = fused_documents[:TOP_K_CONTEXT]
            context_str = "\n\n---\n\n".join(
                [f"Source Document ID: {doc.get('id', 'N/A')}\n\n{doc['contents']}" for doc in context_docs])
            retrieved_ids = [doc.get('id') for doc in context_docs]

            # 3. GENERATION
            prompt_template = """
            You are an expert squash coach AI. Your task is to generate a detailed and coherent squash session plan.

            Generate the plan based *only* on the information provided in the "CONTEXT" section below. Do not add any exercises or information not present in the context.

            The user's request is: "{query}"

            Strictly adhere to all constraints mentioned in the user's request, such as duration, player level, and shot focus. The final output must be a single, complete, and well-structured session plan.

            CONTEXT:
            {context}

            FINAL SQUASH SESSION PLAN:
            """

            final_prompt = prompt_template.format(query=query_text, context=context_str)
            generated_plan = llm_generator.generate(final_prompt, LLM_TEMPERATURE)

            # 4. STORE RESULTS in the specified format
            all_generated_results.append({
                "case_id": f"{grammar}_{CORPUS_SIZE}_{query_id}_k{TOP_K_CONTEXT}",
                "session_id": f"generated_session_{session_counter:02d}",  # A new unique ID for the generated session
                "rag_pipeline": "dynamic_hybrid_rrf_simple_prompt",
                "query_text": query_text,
                "retrieved_documents_info": retrieved_ids,
                "generated_plan": generated_plan,
                "reference": ""
            })
            session_counter += 1

    # Generate a timestamp for the experiment run
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # --- Save all results to a single JSON file ---
    output_filename = f"evaluation_sessions_set_k{TOP_K_CONTEXT}_size{CORPUS_SIZE}_{experiment_timestamp}.json"
    output_path = PROJECT_ROOT / "experiments" /output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Saving all generated plans to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_generated_results, f, indent=2)
    print("✅ JSON file saved successfully.")
