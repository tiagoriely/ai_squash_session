# run_field_retrieval.py

import json
from pathlib import Path
import yaml
import argparse

# --- Core Imports from your RAG library ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.parsers.user_query_parser import parse_user_prompt as _parse_user_prompt # Keep for parser test
from rag.utils import load_and_format_config

# --- 1. Import your new adapter ---
from field_adapters.squash_new_corpus_adapter import SquashNewCorpusAdapter


def load_knowledge_base(corpus_path: str) -> list[dict]:
    """Loads the corpus from a JSONL file."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_field_evaluation(grammar_type: str):
    """Runs a targeted evaluation of the FieldRetriever."""
    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    base_config = load_and_format_config(str(config_path))
    corpus_size = base_config['corpus_size']
    template_context = {'grammar_type': grammar_type.replace('_grammar', ''), 'corpus_size': corpus_size}
    formatted_config = load_and_format_config(str(config_path), template_context)

    corpus_path = project_root / formatted_config['corpus_path']
    field_config_path = project_root / "configs" / "retrieval" / "raw_squash_field_retrieval_config.yaml"

    raw_knowledge_base = load_knowledge_base(str(corpus_path))

    # --- 2. Instantiate adapter and transform the knowledge base ---
    print("Transforming raw knowledge base into canonical format...")
    adapter = SquashNewCorpusAdapter()
    canonical_kb = [adapter.transform(doc) for doc in raw_knowledge_base]
    print(f"Transformation complete. {len(canonical_kb)} documents processed.")

    # Create a list of unique durations from the *canonical* data.
    all_durations = sorted(list(set(doc['duration'] for doc in canonical_kb if 'duration' in doc and doc['duration'] is not None)))

    # --- 3. Initialize the retriever with the CLEAN, CANONICAL data ---
    field_retriever = FieldRetriever(
        knowledge_base=canonical_kb, # Use the transformed data
        config_path=str(field_config_path)
    )

    # --- Define Test Queries ---
    test_queries = {
        "participants": "a solo practice session",
        "level": "a session for advanced players",
        "duration": "show me a 45 minute session",
        "shots": "a drill with cross-court drives",
        "complex_1": "a 60-minute solo session for intermediate players",
        "complex_2": "an advanced two-player session focusing on volley drops and boasting"
    }

    print("=" * 50)
    print("        FIELD RETRIEVER STANDALONE EVALUATION")
    print("=" * 50)

    # --- Run Evaluation Loop ---
    for name, query in test_queries.items():
        print(f"\n\n--- Testing Query: '{name}' ---")
        print(f"   Query Text: '{query}'")

        # --- a) Test the Parser ---
        parsed_desires = _parse_user_prompt(
            query,
            allowed_durations=all_durations
        )
        print("\n   [Parser Output]:")
        for key, value in parsed_desires.items():
            print(f"     - {key}: {value}")

        # --- b) Test the Retriever ---
        # The retriever now works on the clean `canonical_kb` data
        retrieved_results = field_retriever.search(query=query, top_k=3)

        print("\n   [Retriever Top 3 Results]:")
        if not retrieved_results:
            print("     - No matching documents found.")
        else:
            for i, doc in enumerate(retrieved_results):
                # Now we can reliably get canonical keys
                doc_id = doc.get('id')
                score = doc.get('field_score')
                participants = doc.get("participants")
                level = doc.get("squashLevel")
                duration = doc.get("duration")
                shots = doc.get("primaryShots", []) # Or 'shots' depending on what you want to see

                print(f"     {i + 1}. ID: {doc_id} (Score: {score:.2f})")
                print(f"         - Participants: {participants}, Level: {level}, Duration: {duration}")
                print(f"         - Primary Shots: {shots[:5]}...") # Display first 5 for brevity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a standalone evaluation of the FieldRetriever.")
    parser.add_argument(
        "grammar",
        type=str,
        choices=['balanced_grammar', 'high_constraint_grammar', 'loose_grammar'],
        help="The grammar whose corpus you want to test against."
    )
    args = parser.parse_args()
    run_field_evaluation(grammar_type=args.grammar)