# In evaluation/retrieval/evaluate_retrievers.py
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from pathlib import Path

# --- Core Imports from your RAG library ---
from rag.retrieval_fusion import query_aware_fusion
from .utils import load_knowledge_base, load_all_query_sets, initialise_retrievers
from rag.utils import load_and_format_config

# Environment variables for preventing deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Constants ---
# Retrieve more candidates initially to give the fusion process more to work with
CANDIDATE_TOP_K = 30
# The final number of results to keep after fusion
FINAL_TOP_K = 10


def run_evaluation(grammar_type: str):
    """
    Main function to run a comprehensive evaluation pipeline for a given grammar,
    testing standalone retrievers AND the hybrid fusion strategy.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Dynamically get corpus size from config
    semantic_config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    base_semantic_config = load_and_format_config(str(semantic_config_path))
    corpus_size = base_semantic_config['corpus_size']
    template_context = {'grammar_type': grammar_type.replace('_grammar', ''), 'corpus_size': corpus_size}
    temp_config = load_and_format_config(str(semantic_config_path), template_context)
    corpus_path = project_root / temp_config['corpus_path']

    # 1. Load data, queries, and initialise retrievers
    knowledge_base = load_knowledge_base(str(corpus_path))
    queries = load_all_query_sets(project_root, grammar_type, corpus_size)
    retrievers = initialise_retrievers(grammar_type, knowledge_base, project_root, corpus_size)

    all_results = []

    print(f"\nRunning comprehensive evaluation for {len(queries)} queries...")
    for query_info in tqdm(queries, desc="Processing All Queries"):
        query_text = query_info['text']

        # --- Stage 1: Run each standalone retriever ---
        standalone_results_map = {}
        for name, retriever in retrievers.items():
            try:
                retrieved_docs = retriever.search(query=query_text, top_k=CANDIDATE_TOP_K)
                standalone_results_map[name] = retrieved_docs

                # Append the standalone results for analysis (up to final_top_k)
                for rank, doc in enumerate(retrieved_docs[:FINAL_TOP_K]):
                    score_keys = ['semantic_score', 'sparse_score', 'field_score']
                    score = next((doc[key] for key in score_keys if key in doc), 0.0)
                    all_results.append({
                        'query_id': query_info['query_id'],
                        'query_type': query_info['type'],
                        'query_text': query_text,
                        'retriever_name': f"{name}_{grammar_type}",
                        'rank': rank + 1,
                        'document_id': doc.get('id') or doc.get('session_id'),
                        'score': score
                    })
            except Exception as e:
                print(f"\nERROR with standalone retriever '{name}': {e}")
                standalone_results_map[name] = []

        # --- Stage 2: Run the hybrid fusion strategy using the results from Stage 1 ---
        try:
            fused_docs = query_aware_fusion(ranked_lists_map=standalone_results_map, query=query_text)

            # Append the hybrid results for analysis
            for rank, doc in enumerate(fused_docs[:FINAL_TOP_K]):
                all_results.append({
                    'query_id': query_info['query_id'],
                    'query_type': query_info['type'],
                    'query_text': query_text,
                    'retriever_name': f"hybrid_fusion_{grammar_type}",
                    'rank': rank + 1,
                    'document_id': doc.get('id') or doc.get('session_id'),
                    'score': doc.get('fusion_score')
                })
        except Exception as e:
            print(f"\nERROR during hybrid fusion: {e}")

    # --- Stage 3: Save all consolidated results ---
    results_df = pd.DataFrame(all_results)
    output_dir = project_root / "evaluation" / "retrieval" / grammar_type / f"corpus_size_{corpus_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"retrieval_results_all-sets_{grammar_type}_{corpus_size}.csv"

    print(f"\nSaving all results to {output_filename}...")
    results_df.to_csv(output_filename, index=False)
    print("\n✅ Comprehensive evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a comprehensive retriever evaluation for a specific grammar.")
    parser.add_argument(
        "grammar",
        type=str,
        choices=['balanced_grammar', 'high_constraint_grammar', 'loose_grammar'],
        help="The type of grammar to evaluate."
    )
    args = parser.parse_args()
    run_evaluation(grammar_type=args.grammar)