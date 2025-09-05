# scripts/analyse_field_retriever.py

import yaml
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Ensure the project root is in the Python path to find the 'rag' module
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import FieldRetriever and the adapter
from rag.retrieval.field_retriever import FieldRetriever
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Helper function to load a .jsonl file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# New function to get the FieldRetriever
def get_field_retriever() -> FieldRetriever:
    """Sets up the FieldRetriever for analysis."""
    # 1. Load raw corpus from a defined path
    corpus_path = PROJECT_ROOT / "data/processed/balanced_grammar/balanced_10.jsonl"
    raw_corpus = load_jsonl(corpus_path)

    # 2. Adapt the corpus to the flat structure the FieldRetriever expects
    adapter = SquashNewCorpusAdapter()
    adapted_corpus = [adapter.transform(doc) for doc in raw_corpus]

    # 3. Define the config path for the retriever
    config_path = PROJECT_ROOT / "configs/retrieval/raw_squash_field_retrieval_config.yaml"

    # 4. Initialise the retriever with the ADAPTED corpus
    return FieldRetriever(knowledge_base=adapted_corpus, config_path=config_path)


def analyse_query(retriever: FieldRetriever, query_text: str, query_id: str, query_type: str) -> Dict:
    """Runs a query, prints results, and returns calculated metrics."""
    print("=" * 80)
    print(f"ANALYSING QUERY ID: {query_id} ({query_type})")
    print(f"QUERY TEXT: \"{query_text}\"")
    print("-" * 80)

    results = retriever.search(query=query_text, top_k=10)

    # Look for 'field_score' instead of 'sparse_score'
    scores = np.array([doc.get('field_score', 0.0) for doc in results])

    # Handle case where no results are returned (e.g., for irrelevant queries)
    if len(results) == 0:
        print("No results returned (Correct for irrelevant queries).")
        scores = np.array([0.0])  # Create a dummy score array for metrics
        top_doc_id = 'N/A'
    else:
        top_doc_id = results[0].get('id', 'N/A')

    # --- Print Ranked List ---
    print(f"{'Rank':<5} | {'Doc ID':<12} | {'Field Score':<15}")
    print("-" * 45)
    for i, doc in enumerate(results):
        rank = i + 1
        doc_id = doc.get('id', 'N/A')
        score = scores[i]
        print(f"{rank:<5} | {doc_id:<12} | {score:<15.4f}")

    # --- Calculate and Print Metrics ---
    metrics = {
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_score': np.mean(scores),
        'std_dev': np.std(scores),
        'q25': np.quantile(scores, 0.25),
        'median': np.quantile(scores, 0.5),
        'q75': np.quantile(scores, 0.75),
        'top_1_delta': scores[0] - scores[1] if len(scores) > 1 else 0.0,
        'top_doc_id': top_doc_id
    }

    print("\n--- Metrics ---")
    print(f"  Max Score       : {metrics['max_score']:.4f}")
    # ... (rest of print statements) ...
    print(f"  Top-1 Delta (R1-R2): {metrics['top_1_delta']:.4f}  <-- Confidence Metric")
    print("\n")

    return {
        'query_id': query_id,
        'query_type': query_type,
        'query_text': query_text,
        **metrics
    }


if __name__ == "__main__":
    # Use the new initialisation function
    field_retriever = get_field_retriever()

    # query list
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

    # All groups
    all_query_groups = {
        "Complexity Type 1 (Relevant)": high_relevance_complex_1,
        "Complexity Type 2 (Relevant)": high_relevance_complex_2,
        "Relevant (Outside Corpus)": high_relevance_not_in_corpus_queries,
        "Relevant (Other Duration)": high_relevance_OOD_duration,
        "Relevant (Single Shotside)": high_relevance_single_shotside,
        "Vague But Relevant": vague_relevant_queries,
        "Out-of-Scope (Other Sport)": other_sport_queries,
        "Out-of-Scope (Informational)": informational_squash_queries
    }

    results_for_csv = []
    for query_type, queries in all_query_groups.items():
        for query in queries:
            # Pass the field_retriever instance
            row_data = analyse_query(field_retriever, query["text"], query["query_id"], query_type)
            results_for_csv.append(row_data)

    # Update the output filename
    output_path = PROJECT_ROOT / "field_retriever_metrics.csv"
    print("=" * 80)
    print(f"Writing analysis results to: {output_path}")

    if results_for_csv:
        fieldnames = results_for_csv[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_for_csv)
        print("CSV file written successfully.")
    else:
        print("No results to write.")