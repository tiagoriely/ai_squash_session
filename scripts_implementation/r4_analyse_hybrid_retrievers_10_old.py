# scripts/analyse_hybrid_retrievers.py

import yaml
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Import all retrievers and fusion logic ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter
from rag.utils import load_and_format_config

# Import your custom fusion strategies
from rag.retrieval_fusion.strategies import dynamic_query_aware_rrf, standard_unweighted_rrf, static_weighted_rrf

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

    print("✅ All retrievers initialised.")
    return {
        "field_metadata": field_retriever,
        "sparse_bm25": sparse_retriever,
        "semantic_e5": dense_retriever
    }


def analyse_hybrid_strategy(
    standalone_results: Dict[str, List[Dict]],
    query_text: str,
    query_id: str,
    query_type: str,
    strategy_name: str,
    field_scoring_config: Dict
) -> Dict:
    """Applies a specific hybrid strategy and calculates its performance metrics."""
    print("-" * 40)
    print(f"Testing Strategy: {strategy_name} | Query ID: {query_id}")

    fused_results = []
    # --- Apply the selected fusion strategy by calling the correct function ---
    if strategy_name == 'static_weighted_rrf':
        fused_results = static_weighted_rrf(standalone_results)

    elif strategy_name == 'standard_unweighted_rrf':
        fused_results = standard_unweighted_rrf(standalone_results)

    elif strategy_name == 'dynamic_query_aware_rrf':
        fused_results = dynamic_query_aware_rrf(standalone_results, query_text, field_scoring_config)

    # --- Calculate metrics for the fused list ---
    scores = np.array([doc.get('fusion_score', 0.0) for doc in fused_results])
    top_doc_id = fused_results[0].get('id', 'N/A') if fused_results else 'N/A'

    metrics = {
        'max_score': np.max(scores) if scores.size > 0 else 0.0,
        'min_score': np.min(scores) if scores.size > 0 else 0.0,
        'mean_score': np.mean(scores) if scores.size > 0 else 0.0,
        'std_dev': np.std(scores) if scores.size > 0 else 0.0,
        'top_1_delta': scores[0] - scores[1] if len(scores) > 1 else 0.0,
        'top_doc_id': top_doc_id
    }

    print(
        f"  -> Top Doc: {metrics['top_doc_id']}, Max Score: {metrics['max_score']:.4f}, Delta: {metrics['top_1_delta']:.4f}")

    return {'strategy_name': strategy_name, 'query_id': query_id, 'query_type': query_type, **metrics}


if __name__ == "__main__":
    retrievers = get_all_retrievers()

    # Load field scoring
    field_config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    with open(field_config_path, "r", encoding="utf-8") as f:
        # Load the specific dictionary needed by the fusion strategy
        field_scoring_config = yaml.safe_load(f).get("FIELD_SCORING_CONFIG", {})

    high_relevance_complex_1 = [
        {"query_id": "complex_01_cg", "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_02_cg", "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_03_cg",
         "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"},
    ]

    high_relevance_complex_2 = [
        {"query_id": "complex_21_mix", "text": "a 60-minute mix session"},
        {"query_id": "complex_22_mix", "text": "a 60-minute mix session for an intermediate player"},
        {"query_id": "complex_23_mix",
         "text": "a 60-minute mix session for an intermediate player focusing on straight lob"},
        {"query_id": "complex_24_mix",
         "text": "a 60-minute mix session for an intermediate player focusing on forehand straight kill"},
    ]

    high_relevance_not_in_corpus_queries = [
        {"query_id": "ooc_01", "text": "a session focusing on the volley cross"},
        {"query_id": "ooc_02", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_03", "text": "a drill session to improve on the cross-court nick"},
        {"query_id": "ooc_05", "text": "practice the 3-step ghosting"},
        {"query_id": "ooc_06", "text": "a solo to practice cross drops"}

    ]

    high_relevance_OOD_duration = [
        {"query_id": "duration_01",
         "text": "a 90-minute drill session for an advanced player focusing on 2-wall boast"},
        {"query_id": "duration_02",
         "text": "a 75-minute drill session for a professional player focusing on counter drop"},
        {"query_id": "duration_03", "text": "a 30-minute drill session for a intermediate player on straight drop"},

    ]

    high_relevance_single_shotside = [
        {"query_id": "shotside_01", "text": "a 45-minute drill only focusing on backhand side"},
        {"query_id": "shotside_02", "text": "a 60-minute conditioned game only focusing on forehand side"},
    ]

    vague_relevant_queries = [
        {"query_id": "vague_01", "text": "generate a squash session"},
        {"query_id": "vague_02", "text": "generate a session to improve my forehand"},
        {"query_id": "vague_03", "text": "generate a session to work on my movement to the front"},
        {"query_id": "vague_04", "text": "a drill for squash"},
        {"query_id": "vague_05", "text": "a conditioned game session"},

    ]

    other_sport_queries = [
        {"query_id": "other_sport_01", "text": "a drill to improve my tennis serve"},
        {"query_id": "other_sport_02", "text": "how to practice a badminton drop shot"},
        {"query_id": "other_sport_03", "text": "a good warm-up for playing padel"}
    ]

    informational_squash_queries = [
        {"query_id": "squash_other_01", "text": "what is the best squash racket for a beginner"},
        {"query_id": "squash_other_02", "text": "rules of a tie-break in squash"},
        {"query_id": "squash_other_03", "text": "who is currently the best squash player"}
    ]

    random_queries = [
        {"query_id": "random_01", "text": "I love fluffy puppies"},
        {"query_id": "random_02", "text": "Eating apples rocks!"},
        {"query_id": "random_03", "text": "I am the king of Norway"},
        {"query_id": "random_04",
         "text": "Can you recommend a good recipe for beef wellington and suggest a wine pairing?"},
        {"query_id": "random_05",
         "text": "What is the nature of consciousness and how does it relate to the concept of subjective reality?"},
        {"query_id": "random_05",
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

    all_hybrid_results = []
    strategies_to_test = ['static_weighted_rrf', 'standard_unweighted_rrf', 'dynamic_query_aware_rrf']

    for query_type, queries in all_query_groups.items():
        for query in queries:
            # --- Get results from each standalone retriever ONCE per query ---
            standalone_results_map = {}
            for name, retriever in retrievers.items():
                # Note: The keys here ('field_metadata', etc.) must match what the fusion functions expect
                standalone_results_map[name] = retriever.search(query=query['text'], top_k=30)

            # --- Test each hybrid strategy on this set of results ---
            for strategy in strategies_to_test:
                row_data = analyse_hybrid_strategy(
                    standalone_results=standalone_results_map,
                    query_text=query['text'],
                    query_id=query['query_id'],
                    query_type=query_type,
                    strategy_name=strategy,
                    field_scoring_config=field_scoring_config
                )
                all_hybrid_results.append(row_data)

    # --- Write all results to a single CSV file ---
    output_path = PROJECT_ROOT / "hybrid_retrievers_metrics.csv"
    print("\n" + "=" * 80)
    print(f"Writing all hybrid analysis results to: {output_path}")

    if all_hybrid_results:
        fieldnames = all_hybrid_results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_hybrid_results)
        print("✅ Hybrid analysis complete. CSV file written successfully.")