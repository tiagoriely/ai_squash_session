# run:  python3 -m rag.pipelines.retrieval.field_retrieval.run_field_retrieval

import sys
import json
import argparse
from pathlib import Path

# Import the logic (parse_user_prompt in field_matcher delegates to user_desires)
from rag_old.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document
from rag_old.pipelines.retrieval.field_retrieval.adapters import get_adapter
from rag_old.pipelines.retrieval.field_retrieval.user_desires import parse_user_prompt
from rag_old.pipelines.retrieval.field_retrieval.dialogue_manager import refine_user_desires




from evaluation.retrieval.field_fulfilment import best_hit, mean_k, coverage_k
from evaluation.retrieval.field_fulfilment_with_score_weights import best_hit_w, mean_k_w, coverage_k_w

CORPUS_PATHS = {
    "squash_old": Path("data/processed/my_kb.jsonl"),
    "squash_new": Path("data/processed/balanced_grammar/balanced_500.jsonl"),
}


def main():
    parser = argparse.ArgumentParser(description="Run field-based retrieval with a specified corpus adapter.")
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        choices=CORPUS_PATHS.keys(),
        help="The name of the corpus to use."
    )
    args = parser.parse_args()

    # Select the adapter and KB Path based on user input
    adapter = get_adapter(args.corpus)
    KB_PATH = CORPUS_PATHS[args.corpus]

    if not KB_PATH.exists():
        print(f"Error: Knowledge base file not found at {KB_PATH}. Please run corpus_tools.py first.")
        sys.exit(1)

    # Load knowledge base
    raw_knowledge_base = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw_knowledge_base.append(json.loads(line))

    # Allow passing a prompt on the command line; otherwise use the sample
    user_prompt = (
        "Create a 60 minutes conditioned game for an advanced player that helps with "
        "'Strategic Application of Both 2-Wall and 3-Wall Boasts within a Driving Game"
    )

    # Snap durations to common buckets for the initial parse
    allowed_durations = [45, 60, 90]
    initial_desires = parse_user_prompt(user_prompt, allowed_durations=allowed_durations)
    print(f"\nInitial Desires: {initial_desires}")

    # call dialogue manager
    final_user_desires = refine_user_desires(initial_desires)
    print(f"Final User Desires: {final_user_desires}")

    # Score documents
    scored_documents = []
    for raw_doc in raw_knowledge_base:
        # transform the doc before scoring
        canonical_doc = adapter.transform(raw_doc)
        s = score_document(canonical_doc, final_user_desires)
        scored_documents.append((s, canonical_doc))  # Store the canonical doc

    # Rank and take top-N with positive score
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    top_n_documents = [doc for score, doc in scored_documents if score > 0][:5]

    # Print results using the canonical document structure
    print("\n--- Top Relevant Documents (Field Retrieval) ---")
    if top_n_documents:
        for i, doc in enumerate(top_n_documents, start=1):
            # Find the score associated with the canonical doc
            score = next(s for s, d in scored_documents if d["id"] == doc["id"])
            print(f"Rank {i} (Score: {score:.2f}):")
            # Now all keys are consistent and canonical!
            print(f"  ID: {doc.get('id')}, Source: {doc.get('source')}")
            print(
                f"  Type: {doc.get('type')}, Participants: {doc.get('participants')}, Level: {doc.get('squashLevel')}")
            print(f"  Intensity: {doc.get('intensity')}, Duration: {doc.get('duration')}")
            print(f"  Shots: {doc.get('shots')}")
            print(f"  Primary shots: {doc.get('primaryShots')}")
            print("-" * 20)
    else:
        print("No relevant documents found for the given prompt.")

    # Unweighted fulfilment metrics
    if top_n_documents:
        print("\n=== Field-Fulfilment metrics ===")
        print(f"Best-Hit (rank-1)        : {best_hit(top_n_documents, final_user_desires):.2%}")
        print(f"Mean fulfilment top-5    : {mean_k(top_n_documents, final_user_desires, k=5):.2%}")
        print(f"Coverage of req fields@5 : {coverage_k(top_n_documents, final_user_desires, k=5):.2%}")
    else:
        print("Cannot compute fulfilment metrics: no relevant documents.")

    # Weighted fulfilment metrics
    if top_n_documents:
        print("\n=== Weighted Field-Fulfilment metrics with Market Perception ===")
        print("Weighted best-hit:", best_hit_w(top_n_documents, final_user_desires))
        print("Weighted mean-5 :", mean_k_w(top_n_documents, final_user_desires))
        print("Weighted cov@5  :", coverage_k_w(top_n_documents, final_user_desires))
    else:
        print("\n=== Weighted Field-Fulfilment metrics with Market Perception ===")
        print("Cannot compute weighted metrics: no relevant documents.")


if __name__ == "__main__":
    main()
