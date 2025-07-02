# pipelines/hybrid_retriever.py

# ── Open-MP hot-patch ──────────────────────────────────────────────────────────
import os, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # let libomp load twice
os.environ.setdefault("OMP_NUM_THREADS", "1")          # be gentle with threads
torch.set_num_threads(1)
# ───────────────────────────────────────────────────────────────────────────────

import argparse
import time
import json
from pathlib import Path
import yaml

# --- Semantic Retrieval Components (using flashrag from third_party) ---
import torch
import torch.nn as nn

# Monkey-patch torch.cuda if not available, as per your semantic retrieval script
if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self

# CORRECTED IMPORT PATH FOR DenseRetriever
from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever

# --- Field Retrieval Components ---
# Import from your new field_matcher.py
from .field_matcher import parse_user_prompt, score_document


# Function to load YAML config (copied from your semantic retrieval script)
def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Hybrid Retrieval Logic ---

# THIS IS THE ONE AND ONLY hybrid_search FUNCTION DEFINITION
def hybrid_search(user_query: str,
                  knowledge_base_docs: list[dict],  # Pass the full loaded KB
                  semantic_retriever: DenseRetriever,
                  semantic_threshold: float = 0.5,  # Minimum semantic similarity score
                  final_top_k: int = 5):
    """
    Performs a hybrid search combining field-based matching and semantic similarity.

    Args:
        user_query (str): The user's input query.
        knowledge_base_docs (list[dict]): All documents loaded from your my_kb.jsonl.
        semantic_retriever (DenseRetriever): An initialized flashrag DenseRetriever instance.
        semantic_threshold (float): Minimum semantic similarity score for a document to be considered.
        final_top_k (int): The number of final top documents to return after all filters.

    Returns:
        list[dict]: A list of the most relevant documents, including their scores.
    """

    # 1. Parse User Prompt for Field Desires
    user_desires = parse_user_prompt(user_query)
    print(f"\nUser Desires: {user_desires}")

    # 2. Perform Semantic Search to get initial candidates and their scores
    print("\n--- Performing Semantic Retrieval (via flashrag) ---")
    t0_sem_search = time.perf_counter()

    # Initialize sem_docs_and_scores to ensure it's always defined
    sem_docs_and_scores = ([], []) # <--- Moved this line here and ensures it's always present

    try:
        # flashrag's DenseRetriever.search will handle querying the index
        # It should use the topk from the config passed during initialization
        sem_docs_and_scores = semantic_retriever.search(user_query, return_score=True)
    except Exception as e:
        print(f"Error during semantic search: {e}. Semantic results will be empty.")
        # sem_docs_and_scores remains ([], []) if an error occurs

    t1_sem_search = time.perf_counter()
    print(f"Semantic search took {1000 * (t1_sem_search - t0_sem_search):.1f} ms.")

    # Convert flashrag results into a dictionary for quick lookup by ID
    # This line is now at the correct level and will have sem_docs_and_scores defined
    sem_score_map = {doc['id']: score for doc, score in zip(sem_docs_and_scores[0], sem_docs_and_scores[1])}

    # Create a mapping from document ID to the full document object from the initial KB load
    doc_lookup_map = {doc['id']: doc for doc in knowledge_base_docs}

    # 3. Apply Semantic Threshold and Calculate Field Scores
    candidates = []
    print(f"\n--- Applying Semantic Threshold ({semantic_threshold:.2f}) & Calculating Field Scores ---")
    for doc_id, sem_score in sem_score_map.items():
        doc = doc_lookup_map.get(doc_id)
        if doc is None:
            # This should ideally not happen if doc_id_map is complete
            print(f"Warning: Document with ID {doc_id} from semantic search not found in loaded knowledge base.")
            continue

        if sem_score >= semantic_threshold:
            field_score = score_document(doc, user_desires)
            candidates.append({
                'doc': doc,
                'field_score': field_score,
                'semantic_score': sem_score,
                # Use field_score as the primary ranking metric for now
                'final_ranking_score': field_score
            })
        else:
            print(
                f"Doc ID {doc.get('id', 'N/A')} (Source: {doc.get('source', 'N/A')}) filtered out (Semantic Score: {sem_score:.2f} < Threshold {semantic_threshold:.2f})")

    # 4. Rank Candidates by Final Ranking Score (Field Score in this case)
    candidates.sort(key=lambda x: x['final_ranking_score'], reverse=True)

    # 5. Final Top-K Selection
    final_results = candidates[:final_top_k]

    return final_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid Retriever for Squash Sessions")
    ap.add_argument("--query", required=True, help="User query for a squash session.")
    ap.add_argument("--sem_threshold", type=float, default=0.5,
                    help="Minimum semantic similarity score to include a document. (Default: 0.5)")
    ap.add_argument("--final_topk", type=int, default=5,
                    help="Number of final top documents to return. (Default: 5)")
    ap.add_argument("--retriever_cfg", default="configs/retrieval/faiss_rerank.yaml",
                    help="Path to semantic retriever YAML config.")
    args = ap.parse_args()

    # Load the YAML config file into a dictionary BEFORE passing to DenseRetriever
    retriever_config_dict = load_cfg(args.retriever_cfg)

    # Load Knowledge Base (corpus_path from flashrag config)
    KB_PATH = Path(retriever_config_dict['corpus_path'])

    if not KB_PATH.exists():
        print(
            f"Error: Knowledge base corpus file not found at {KB_PATH}. Please ensure corpus_tools.py has run and corpus_path in config is correct.")
        exit()

    knowledge_base = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            knowledge_base.append(json.loads(line))
            # Removed the warning about missing embeddings as it's no longer relevant
            # for flashrag's DenseRetriever, which uses the FAISS index.

    # Initialize Semantic Retriever using flashrag's DenseRetriever and the loaded config
    print(f"\n--- Initializing flashrag DenseRetriever with config: {args.retriever_cfg} ---")
    semantic_retriever = DenseRetriever(retriever_config_dict)

    t0_total = time.perf_counter()
    results = hybrid_search(
        user_query=args.query,
        knowledge_base_docs=knowledge_base,
        semantic_retriever=semantic_retriever,
        semantic_threshold=args.sem_threshold,
        final_top_k=args.final_topk
    )
    t1_total = time.perf_counter()

    print(f"\n--- Hybrid Retrieval Results (Total Time: {1000 * (t1_total - t0_total):.1f} ms) ---")
    if results:
        for i, res in enumerate(results):
            doc = res['doc']
            print(f"Rank {i + 1}:")
            print(f"  ID: {doc.get('id')}, Source: {doc.get('source')}")
            print(f"  Field Score: {res['field_score']:.2f}, Semantic Score: {res['semantic_score']:.4f}")
            print(
                f"  Type: {doc.get('type')}, Participants: {doc.get('participants')}, Level: {doc.get('squashLevel')}")
            print(f"  Intensity: {doc.get('intensity')}, Duration: {doc.get('duration')}")
            print(f"  Shots: {doc.get('shots')}, Shot Side: {doc.get('shotSide')}")
            print(f"  Spec. Shots: {doc.get('specificShots')}")
            print("-" * 20)
    else:
        print("No documents found meeting the hybrid criteria.")