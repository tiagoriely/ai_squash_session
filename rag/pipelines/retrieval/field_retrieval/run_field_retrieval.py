# run:  python3 -m rag.pipelines.retrieval.field_retrieval.run_field_retrieval

import re
import json
from pathlib import Path
# Import the logic, and also the clean_and_standardise_value function for evaluation
from rag.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document, clean_and_standardise_value
from evaluation.retrieval.field_fulfilment import best_hit, mean_k, coverage_k
from evaluation.retrieval.field_fulfilment_with_score_weights import (
    best_hit_w, mean_k_w, coverage_k_w
)



# Example Usage and Retrieval Logic (for demonstration purposes)
if __name__ == "__main__":
    KB_PATH = Path("data/my_kb.jsonl")

    if not KB_PATH.exists():
        print(f"Error: Knowledge base file not found at {KB_PATH}. Please run corpus_tools.py first.")
        exit()

    # Load your knowledge base
    knowledge_base = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            knowledge_base.append(json.loads(line))

    user_prompt = "Create a training routine that helps with 'Strategic Application of Both 2-Wall and 3-Wall Boasts within a Driving Game"
    print(f"User prompt: {user_prompt}")
    user_desires = parse_user_prompt(user_prompt)
    print(f"\nUser Desires: {user_desires}")

    scored_documents = []
    for doc in knowledge_base:
        s = score_document(doc, user_desires)
        scored_documents.append((s, doc))

    scored_documents.sort(key=lambda x: x[0], reverse=True)
    top_n_documents = [doc for score, doc in scored_documents if score > 0][:5]

    print("\n--- Top Relevant Documents (Field Retrieval) ---")
    if top_n_documents:
        for i, doc in enumerate(top_n_documents):
            score = next(s for s, d in scored_documents if d['id'] == doc['id'])
            print(f"Rank {i+1} (Score: {score:.2f}):")
            print(f"  ID: {doc.get('id')}, Source: {doc.get('source')}")
            print(f"  Type: {doc.get('type')}, Participants: {doc.get('participants')}, Level: {doc.get('squashLevel')}")
            print(f"  Intensity: {doc.get('intensity')}, Duration: {doc.get('duration')}")
            print(f"  Shots: {doc.get('shots')}, Shot Side: {doc.get('shotSide')}")
            print(f"  Primary shots: {doc.get('primaryShots')}")
            print(f"  Secondary shots: {doc.get('secondaryShots')}")
            print("-" * 20)
    else:
        print("No relevant documents found for the given prompt.")

    # Automated Evaluation of Generation (Conceptual, using top field-retrieved doc)
    if top_n_documents:
        print("\n=== Field-Fulfilment metrics ===")
        print(f"Best-Hit (rank-1)        : {best_hit(top_n_documents, user_desires):.2%}")
        print(f"Mean fulfilment top-5    : {mean_k(top_n_documents, user_desires, k=5):.2%}")
        print(f"Coverage of req fields@5 : {coverage_k(top_n_documents, user_desires, k=5):.2%}")
    else:
        print("Cannot compute fulfilment metrics: no relevant documents.")


    # PROBLEM HERE
    print("\n=== Weighted Field-Fulfilment metrics with Market Perception ===")
    print("Weighted best‑hit:", best_hit_w(top_n_documents, user_desires))
    print("Weighted mean‑5 :", mean_k_w(top_n_documents, user_desires))
    print("Weighted cov@5  :", coverage_k_w(top_n_documents, user_desires))
