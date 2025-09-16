# scripts/generate_hybrid_results.py

import yaml
import json
import csv
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Import all retrievers and fusion logic ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.utils import load_and_format_config

# Import your custom fusion strategies
from rag.retrieval_fusion.strategies import dynamic_query_aware_rrf, standard_unweighted_rrf, static_weighted_rrf, dynamic_query_aware_score_fusion


def get_all_retrievers() -> Dict[str, Any]:
    """A helper function to initialise all three standalone retrievers."""
    print("Initialising all standalone retrievers...")

    # --- Field Retriever Setup ---
    corpus_path = PROJECT_ROOT / "data/processed/balanced_grammar/balanced_10.jsonl"
    raw_corpus = [json.loads(line) for line in open(corpus_path, 'r', encoding='utf-8')]
    adapter = SquashNewCorpusAdapter()
    adapted_corpus = [adapter.transform(doc) for doc in raw_corpus]
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    field_retriever = FieldRetriever(knowledge_base=adapted_corpus, config_path=field_config_path)

    # --- Metadata Sparse Retriever Setup ---
    sparse_config_path = PROJECT_ROOT / "configs/retrieval/sparse_retriever.yaml"
    context = {"grammar_type": "balanced", "corpus_size": 10}
    sparse_config = load_and_format_config(str(sparse_config_path), context)
    sparse_config['sparse_params']['index_path'] = str(PROJECT_ROOT / sparse_config['sparse_params']['index_path'])
    sparse_retriever = SparseRetriever(knowledge_base=raw_corpus, config=sparse_config['sparse_params'])

    # --- Dense Retriever Setup ---
    dense_config_path = PROJECT_ROOT / "configs/retrieval/semantic_retriever.yaml"
    dense_config = load_and_format_config(str(dense_config_path), context)
    dense_config['corpus_path'] = str(PROJECT_ROOT / dense_config['corpus_path'])
    dense_config['index_path'] = str(PROJECT_ROOT / dense_config['index_path'])
    dense_retriever = SemanticRetriever(config=dense_config)

    return {
        "field_metadata": field_retriever,
        "sparse_bm25": sparse_retriever,
        "semantic_e5": dense_retriever
    }

def run_all_hybrid_strategies(retrievers: Dict, query: Dict, field_scoring_config: Dict):
    """
    Runs a single query through all hybrid strategies and returns the detailed
    top-10 results for each in a "long format".
    """
    standalone_results_map = {
        name: retriever.search(query=query['text'], top_k=30)
        for name, retriever in retrievers.items()
    }

    strategies = {
        'Static Weighted RRF': lambda: static_weighted_rrf(standalone_results_map),
        'Standard Unweighted RRF': lambda: standard_unweighted_rrf(standalone_results_map),
        'Dynamic Query-Aware RRF (Rank)': lambda: dynamic_query_aware_rrf(standalone_results_map, query['text'], field_scoring_config),
        'Dynamic Query-Aware Fusion': lambda: dynamic_query_aware_score_fusion(standalone_results_map, query['text'],
                                                                               field_scoring_config)
    }

    long_format_results = []
    for name, strategy_func in strategies.items():
        print(f"  - Running strategy: {name}")
        fused_results = strategy_func()
        for rank, doc in enumerate(fused_results[:10]): # Get top 10
            long_format_results.append({
                'query_id': query['query_id'],
                'query_type': query['type'],
                'strategy_name': name,
                'rank': rank + 1,
                'doc_id': doc.get('id', 'N/A'),
                'fusion_score': doc.get('fusion_score', 0.0)
            })
    return long_format_results


if __name__ == "__main__":
    retrievers = get_all_retrievers()

    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    with open(field_config_path, "r", encoding="utf-8") as f:
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    # "Complexity Type 1 (Relevant)": high_relevance_complex_1,
    # "Complexity Type 2 (Relevant)": high_relevance_complex_2,
    # "Relevant (Outside Corpus)": high_relevance_not_in_corpus_queries,
    # "Relevant (Other Duration)": high_relevance_OOD_duration,
    # "Relevant (Single Shotside)": high_relevance_single_shotside,
    # "Vague But Relevant": vague_relevant_queries,
    # "Out-of-Scope (Other Sport)": other_sport_queries,
    # "Out-of-Scope (Informational)": informational_squash_queries,
    # "Random (non-Relevant)": random_queries,

    high_relevance_complex_1 = [
        {"query_id": "complex_01_cg", "type": "Complexity Type 1 (Relevant)", "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_02_cg", "type": "Complexity Type 1 (Relevant)", "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_03_cg", "type": "Complexity Type 1 (Relevant)",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
    ]

    high_relevance_complex_2 = [
        {"query_id": "complex_21_mix", "type": "Complexity Type 2 (Relevant)", "text": "a 60-minute mix session"},
        {"query_id": "complex_22_mix", "type": "Complexity Type 2 (Relevant)", "text": "a 60-minute mix session for an intermediate player"},
        {"query_id": "complex_23_mix", "type": "Complexity Type 2 (Relevant)",
         "text": "a 60-minute mix session for an intermediate player focusing on straight lob"},
        {"query_id": "complex_24_mix", "type": "Complexity Type 2 (Relevant)",
         "text": "a 60-minute mix session for an intermediate player focusing on forehand straight kill"},
    ]

    high_relevance_not_in_corpus_queries = [
        {"query_id": "ooc_01", "type": "Relevant (Outside Corpus)", "text": "a session focusing on the volley cross"},
        {"query_id": "ooc_02", "type": "Relevant (Outside Corpus)", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_03", "type": "Relevant (Outside Corpus)", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_05", "type": "Relevant (Outside Corpus)", "text": "practice the 3-step ghosting"},
        {"query_id": "ooc_06", "type": "Relevant (Outside Corpus)", "text": "a solo to practice cross drops"}

    ]

    high_relevance_OOD_duration = [
        {"query_id": "duration_01", "type": "Relevant (Other Duration)",
         "text": "a 90-minute drill session for an advanced player focusing on 2-wall boast"},
        {"query_id": "duration_02", "type": "Relevant (Other Duration)",
         "text": "a 75-minute drill session for a professional player focusing on counter drop"},
        {"query_id": "duration_03", "type": "Relevant (Other Duration)", "text": "a 30-minute drill session for a intermediate player on straight drop"},

    ]

    high_relevance_single_shotside = [
        {"query_id": "shotside_01", "type": "Relevant (Single Shotside)", "text": "a 45-minute drill only focusing on backhand side"},
        {"query_id": "shotside_02", "type": "Relevant (Single Shotside)", "text": "a 60-minute conditioned game only focusing on forehand side"},
    ]

    vague_relevant_queries = [
        {"query_id": "vague_01", "type": "Vague But Relevant", "text": "generate a squash session"},
        {"query_id": "vague_02", "type": "Vague But Relevant", "text": "generate a session to improve my forehand"},
        {"query_id": "vague_03", "type": "Vague But Relevant", "text": "generate a session to work on my movement to the front"},
        {"query_id": "vague_04", "type": "Vague But Relevant", "text": "a drill for squash"},
        {"query_id": "vague_05", "type": "Vague But Relevant", "text": "a conditioned game session"},

    ]

    other_sport_queries = [
        {"query_id": "other_sport_01", "type": "Out-of-Scope (Other Sport)", "text": "a drill to improve my tennis serve"},
        {"query_id": "other_sport_02", "type": "Out-of-Scope (Other Sport)", "text": "how to practice a badminton drop shot"},
        {"query_id": "other_sport_03", "type": "Out-of-Scope (Other Sport)", "text": "a good warm-up for playing padel"}
    ]

    informational_squash_queries = [
        {"query_id": "squash_other_01", "type": "Out-of-Scope (Informational)", "text": "what is the best squash racket for a beginner"},
        {"query_id": "squash_other_02", "type": "Out-of-Scope (Informational)", "text": "rules of a tie-break in squash"},
        {"query_id": "squash_other_03", "type": "Out-of-Scope (Informational)", "text": "who is currently the best squash player"}
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

    all_results_for_csv = []
    for query_type, queries in all_query_groups.items():
        print(f"\nProcessing query type: {query_type}")
        for query in queries:
            print(f"Query: {query['query_id']}")
            query_results = run_all_hybrid_strategies(retrievers, query, field_scoring_config)
            all_results_for_csv.extend(query_results)

    output_path = PROJECT_ROOT / "hybrid_retrievers_detailed_results.csv"
    print("\n" + "=" * 80)
    print(f"Writing detailed hybrid results to: {output_path}")

    if all_results_for_csv:
        fieldnames = all_results_for_csv[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results_for_csv)
        print("✅ Detailed results CSV written successfully.")