# scripts/run_final_evaluation.py

import yaml
import json
import csv
from pathlib import Path
import sys
import os
from tqdm import tqdm

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Core Component Imports ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.utils import load_and_format_config
from rag.retrieval_fusion.strategies import dynamic_query_aware_rrf


def get_retrievers_for_grammar(grammar_type: str, corpus_size: int) -> dict[str, any]:
    """Initialises all three standalone retrievers for a specific grammar and corpus size."""
    print(f"\n--- Initialising retrievers for [{grammar_type.upper()}] grammar (size {corpus_size}) ---")

    context = {"grammar_type": grammar_type, "corpus_size": corpus_size}

    corpus_path_str = f"data/processed/{grammar_type}_grammar/{grammar_type}_{corpus_size}.jsonl"
    corpus_path = PROJECT_ROOT / corpus_path_str
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    raw_corpus = [json.loads(line) for line in open(corpus_path, 'r', encoding='utf-8')]

    # --- Field Retriever ---
    adapter = SquashNewCorpusAdapter()
    adapted_corpus = [adapter.transform(doc) for doc in raw_corpus]
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    field_retriever = FieldRetriever(knowledge_base=adapted_corpus, config_path=field_config_path)

    # --- Metadata Sparse Retriever ---
    sparse_config_path = PROJECT_ROOT / "configs/retrieval/sparse_retriever.yaml"
    sparse_config = load_and_format_config(str(sparse_config_path), context)
    sparse_config['sparse_params']['index_path'] = str(PROJECT_ROOT / sparse_config['sparse_params']['index_path'])
    sparse_retriever = SparseRetriever(knowledge_base=raw_corpus, config=sparse_config['sparse_params'])

    # --- Dense Retriever ---
    dense_config_path = PROJECT_ROOT / "configs/retrieval/semantic_retriever.yaml"
    dense_config = load_and_format_config(str(dense_config_path), context)
    dense_config['corpus_path'] = str(PROJECT_ROOT / dense_config['corpus_path'])
    dense_config['index_path'] = str(PROJECT_ROOT / dense_config['index_path'])
    dense_retriever = SemanticRetriever(config=dense_config)

    print("✅ All retrievers initialised.")
    return {
        "Field Retriever": field_retriever,
        "Metadata Sparse": sparse_retriever,
        "Dense Retriever": dense_retriever
    }


def run_evaluation_for_grammar(grammar_type: str, corpus_size: int, queries: list[dict]):
    """
    Runs the full evaluation for a single grammar type, testing all standalone
    retrievers and the dynamic hybrid strategy. Returns results in a list.
    """

    # --- Setup ---
    retrievers = get_retrievers_for_grammar(grammar_type, corpus_size)
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    with open(field_config_path, "r", encoding="utf-8") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    all_results = []

    # --- Run Evaluations ---
    for query in tqdm(queries, desc=f"Evaluating [{grammar_type.upper()}]"):
        # 1. Get standalone results
        standalone_results_map = {}
        for name, retriever in retrievers.items():
            docs = retriever.search(query=query['text'], top_k=10)
            standalone_results_map[name.replace(" ", "_").lower()] = docs

            # Store detailed results for standalone models
            for rank, doc in enumerate(docs):
                score_key = {'Field': 'field_score', 'Metadata': 'sparse_score', 'Dense': 'semantic_score'}[
                    name.split(' ')[0]]
                all_results.append({
                    'grammar_type': grammar_type,
                    'query_id': query['query_id'],
                    'query_type': query['type'],
                    'strategy_name': name,
                    'rank': rank + 1,
                    'doc_id': doc.get('id', 'N/A'),
                    'score': doc.get(score_key, 0.0)
                })

        # 2. Run Dynamic Hybrid strategy
        # Map to keys expected by fusion function
        fusion_map = {
            'field_metadata': standalone_results_map['field_retriever'],
            'sparse_bm25': standalone_results_map['metadata_sparse'],
            'semantic_e5': standalone_results_map['dense_retriever']
        }
        hybrid_docs = dynamic_query_aware_rrf(fusion_map, query['text'], field_scoring_config)

        # Slice the 'hybrid_docs' list to only loop through the top 10 results
        for rank, doc in enumerate(hybrid_docs[:10]):
            all_results.append({
                'grammar_type': grammar_type,
                'query_id': query['query_id'],
                'query_type': query['type'],
                'strategy_name': 'Dynamic Hybrid RRF',
                'rank': rank + 1,
                'doc_id': doc.get('id', 'N/A'),
                'score': doc.get('fusion_score', 0.0)
            })

    return all_results


if __name__ == "__main__":
    high_relevance_complex_1 = [
        {"query_id": "complex_01_cg", "type": "Complexity Type 1 (Relevant)",
         "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_02_cg", "type": "Complexity Type 1 (Relevant)",
         "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_03_cg", "type": "Complexity Type 1 (Relevant)",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
    ]

    high_relevance_complex_2 = [
        {"query_id": "complex_21_mix", "type": "Complexity Type 2 (Relevant)", "text": "a 60-minute mix session"},
        {"query_id": "complex_22_mix", "type": "Complexity Type 2 (Relevant)",
         "text": "a 60-minute mix session for an intermediate player"},
        {"query_id": "complex_23_mix", "type": "Complexity Type 2 (Relevant)",
         "text": "a 60-minute mix session for an intermediate player focusing on straight lob"},
        {"query_id": "complex_24_mix", "type": "Complexity Type 2 (Relevant)",
         "text": "a 60-minute mix session for an intermediate player focusing on forehand straight kill"},
    ]

    high_relevance_not_in_corpus_queries = [
        {"query_id": "ooc_01", "type": "Relevant (Outside Corpus)", "text": "a session focusing on the volley cross"},
        {"query_id": "ooc_02", "type": "Relevant (Outside Corpus)",
         "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_03", "type": "Relevant (Outside Corpus)",
         "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_05", "type": "Relevant (Outside Corpus)", "text": "practice the 3-step ghosting"},
        {"query_id": "ooc_06", "type": "Relevant (Outside Corpus)", "text": "a solo to practice cross drops"}

    ]

    high_relevance_OOD_duration = [
        {"query_id": "duration_01", "type": "Relevant (Other Duration)",
         "text": "a 90-minute drill session for an advanced player focusing on 2-wall boast"},
        {"query_id": "duration_02", "type": "Relevant (Other Duration)",
         "text": "a 75-minute drill session for a professional player focusing on counter drop"},
        {"query_id": "duration_03", "type": "Relevant (Other Duration)",
         "text": "a 30-minute drill session for a intermediate player on straight drop"},

    ]

    high_relevance_single_shotside = [
        {"query_id": "shotside_01", "type": "Relevant (Single Shotside)",
         "text": "a 45-minute drill only focusing on backhand side"},
        {"query_id": "shotside_02", "type": "Relevant (Single Shotside)",
         "text": "a 60-minute conditioned game only focusing on forehand side"},
    ]

    vague_relevant_queries = [
        {"query_id": "vague_01", "type": "Vague But Relevant", "text": "generate a squash session"},
        {"query_id": "vague_02", "type": "Vague But Relevant", "text": "generate a session to improve my forehand"},
        {"query_id": "vague_03", "type": "Vague But Relevant",
         "text": "generate a session to work on my movement to the front"},
        {"query_id": "vague_04", "type": "Vague But Relevant", "text": "a drill for squash"},
        {"query_id": "vague_05", "type": "Vague But Relevant", "text": "a conditioned game session"},

    ]

    other_sport_queries = [
        {"query_id": "other_sport_01", "type": "Out-of-Scope (Other Sport)",
         "text": "a drill to improve my tennis serve"},
        {"query_id": "other_sport_02", "type": "Out-of-Scope (Other Sport)",
         "text": "how to practice a badminton drop shot"},
        {"query_id": "other_sport_03", "type": "Out-of-Scope (Other Sport)", "text": "a good warm-up for playing padel"}
    ]

    informational_squash_queries = [
        {"query_id": "squash_other_01", "type": "Out-of-Scope (Informational)",
         "text": "what is the best squash racket for a beginner"},
        {"query_id": "squash_other_02", "type": "Out-of-Scope (Informational)",
         "text": "rules of a tie-break in squash"},
        {"query_id": "squash_other_03", "type": "Out-of-Scope (Informational)",
         "text": "who is currently the best squash player"}
    ]

    random_queries = [
        {"query_id": "random_01", "type": "Random (non-Relevant)", "text": "I love fluffy puppies"},
        {"query_id": "random_02", "type": "Random (non-Relevant)", "text": "Eating apples rocks!"},
        {"query_id": "random_03", "type": "Random (non-Relevant)", "text": "I am the king of Norway"},
        {"query_id": "random_04", "type": "Random (non-Relevant)",
         "text": "Can you recommend a good recipe for beef wellington and suggest a wine pairing?"},
        {"query_id": "random_05", "type": "Random (non-Relevant)",
         "text": "What is the nature of consciousness and how does it relate to the concept of subjective reality?"},
        {"query_id": "random_05", "type": "Random (non-Relevant)",
         "text": "SELECT user_id, last_login FROM users WHERE account_status = 'active' AND last_login < NOW() - INTERVAL '30 days';"}
    ]

    # All groups
    all_query_groups = {
        "Complexity Type 1 (Relevant)": high_relevance_complex_1,
        "Complexity Type 2 (Relevant)": high_relevance_complex_2,
        "Relevant (Outside Corpus)": high_relevance_not_in_corpus_queries,
        "Relevant (Other Duration)": high_relevance_OOD_duration,
        "Relevant (Single Shotside)": high_relevance_single_shotside,
        "Vague But Relevant": vague_relevant_queries,
        "Out-of-Scope (Other Sport)": other_sport_queries,
        "Out-of-Scope (Informational)": informational_squash_queries,
        "Random (non-Relevant)": random_queries,
    }

    # Flatten all queries into a single list
    all_queries = [query for group in all_query_groups.values() for query in group]

    # --- Main Experiment Loop ---
    grammar_types_to_test = ['loose', 'high_constraint', 'balanced']
    corpus_size = 100
    final_results = []

    for grammar in grammar_types_to_test:
        try:
            results_for_grammar = run_evaluation_for_grammar(grammar, corpus_size, all_queries)
            final_results.extend(results_for_grammar)
        except FileNotFoundError as e:
            print(f"\nSKIPPING grammar '{grammar}': {e}")
            continue

    # --- Save Consolidated Results ---
    output_path = PROJECT_ROOT / f"final_evaluation_all_grammars_size{corpus_size}.csv"
    print("\n" + "=" * 80)
    print(f"Writing all evaluation results to: {output_path}")

    if final_results:
        fieldnames = final_results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)
        print("✅ Final evaluation complete. CSV file written successfully.")