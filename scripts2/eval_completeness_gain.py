# scripts2/eval_completeness_gain.py
import yaml
import json
from pathlib import Path
import sys
import os
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

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
from evaluation.utils.completeness_gain import CompletenessGain


def initialise_components(grammar_type: str, corpus_size: int) -> tuple:
    """A helper function to set up and initialise all necessary RAG components."""
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

    retrievers = {"field_metadata": field_retriever, "sparse_bm25": sparse_retriever, "semantic_e5": dense_retriever}
    generator = Generator(model="gpt-4o")
    print("✅ All components initialised.")
    return retrievers, generator


if __name__ == "__main__":
    # --- Experiment Parameters ---
    TOP_K_CONTEXT = 10
    LLM_TEMPERATURE = 0.2  # Using a low temperature for more deterministic structural output
    GRAMMARS_TO_TEST = ['loose', 'balanced', 'high_constraint']
    SIZES_TO_TEST = [50, 100, 200, 300, 500]

    GRAMMAR_PATHS = {
        "balanced": "grammar/sports/squash/balanced_grammar/loose_structures.ebnf",
        "high_constraint": "grammar/sports/squash/high_constraint_grammar/strict_structures.ebnf",
        "loose": "grammar/sports/squash/loose_grammar/loose_structures.ebnf"
    }

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
        {"query_id": "complex_33_drill",
         "text": "a drill session for a professional to improve on the backhand 3-wall boast"},
    ]

    with open(PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml", "r") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    all_results = []

    for grammar in GRAMMARS_TO_TEST:
        if grammar not in GRAMMAR_PATHS:
            print(f"Warning: No EBNF grammar path defined for '{grammar}'. Skipping.")
            continue
        ebnf_path = PROJECT_ROOT / GRAMMAR_PATHS[grammar]
        completeness_evaluator = CompletenessGain(ebnf_grammar_path=ebnf_path)
        print(f"\n--- Initialised CompletenessGain evaluator for [{grammar.upper()}] using: {ebnf_path.name} ---")

        for size in SIZES_TO_TEST:
            try:
                retrievers, llm_generator = initialise_components(grammar, size)
            except FileNotFoundError as e:
                print(f"\nSKIPPING combination '{grammar}'/'{size}': {e}")
                continue

            desc = f"Evaluating [{grammar.upper()}-{size}]"
            for query_info in tqdm(queries_for_evaluation, desc=desc):
                query_id = query_info["query_id"]
                query_text = query_info["text"]

                # --- Generate BASELINE Plan (NO RAG) ---
                prompt_no_rag = f"You are an expert squash coach AI. Generate a detailed and coherent squash session plan for the following request: \"{query_text}\""
                baseline_plan = llm_generator.generate(prompt_no_rag, temperature=LLM_TEMPERATURE)

                # --- Generate TARGET Plan (WITH RAG) ---
                standalone_results = {name: retriever.search(query=query_text, top_k=30) for name, retriever in
                                      retrievers.items()}
                fused_documents = dynamic_query_aware_score_fusion(standalone_results, query_text, field_scoring_config)
                context_docs = fused_documents[:TOP_K_CONTEXT]
                context_str = "\n\n---\n\n".join(
                    [f"Source Document ID: {doc.get('id', 'N/A')}\n\n{doc.get('contents', '')}" for doc in
                     context_docs])

                prompt_with_rag_template = """
                You are an expert squash coach AI. Your task is to generate a detailed and coherent squash session plan.
    
                Generate the plan based *only* on the information provided in the "CONTEXT" section below. Do not add any exercises or information not present in the context.
    
                The user's request is: "{query}"
    
                Strictly adhere to all constraints mentioned in the user's request, such as duration, player level, and shot focus. The final output must be a single, complete, and well-structured session plan.
    
                CONTEXT:
                {context}
    
                FINAL SQUASH SESSION PLAN:
                """
                prompt_with_rag = prompt_with_rag_template.format(query=query_text, context=context_str)
                target_plan = llm_generator.generate(prompt_with_rag, temperature=LLM_TEMPERATURE)

                scores = completeness_evaluator.get_score(generated_plan=target_plan, baseline_text=baseline_plan)

                all_results.append({
                    "grammar": grammar,
                    "size": size,
                    "query_id": query_id,
                    "query_text": query_text,
                    **scores  # Unpack the scores dictionary into the results
                })

    # --- FINAL SUMMARY ---
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "experiments" / "completeness_gain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw data to JSON
    raw_output_path = output_dir / f"completeness_gain_raw_{experiment_timestamp}.json"
    print("\n" + "=" * 80)
    print(f"Saving all raw evaluation results to: {raw_output_path}")
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("✅ Raw JSON file saved successfully.")

    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_df = results_df.groupby(['grammar', 'size']).mean(numeric_only=True).reset_index()
        summary_df = summary_df.sort_values(by=['grammar', 'size'])

        csv_output_path = output_dir / f"completeness_gain_summary_{experiment_timestamp}.csv"
        print(f"\nSaving summary of results to: {csv_output_path}")
        summary_df.to_csv(csv_output_path, index=False, float_format="%.4f")
        print("✅ Summary CSV file saved successfully.")

        print("\n--- Overall Averages per Grammar/Size ---")
        print(summary_df.to_string(index=False))