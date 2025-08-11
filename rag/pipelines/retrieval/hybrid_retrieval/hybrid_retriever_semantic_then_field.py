# python3 -m rag.pipelines.retrieval.hybrid_retrieval.hybrid_retriever_semantic_then_field \
# --query "I want an advanced drill for 2 players focusing on cross lobs with medium intensity lasting about 45 minutes." \
# --sem_threshold 0.6 \
# --final_topk 5 \
# --alpha 0.7 \
# --retriever_cfg rag/configs/retrieval/faiss_rerank.yaml


import os, torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

import argparse
import time
import json
from pathlib import Path
import yaml
import torch.nn as nn

# Monkey-patch torch.cuda if not available
if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self

from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever
from rag.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


### REFACTORED SCORING LOGIC ###
def combine_scores(candidates: list[dict], alpha: float = 0.7) -> list[dict]:
    """
    Normalizes field scores and calculates a final weighted score.
    """
    if not candidates:
        return []

    # Normalize field_score to a 0-1 range to make it comparable to semantic_score
    max_field_score = max(c['field_score'] for c in candidates)
    if max_field_score > 0:
        for c in candidates:
            c['normalized_field_score'] = c['field_score'] / max_field_score
    else:
        for c in candidates:
            c['normalized_field_score'] = 0.0

    # Calculate the final combined score
    for c in candidates:
        c['final_score'] = (alpha * c['normalized_field_score']) + ((1 - alpha) * c['semantic_score'])

    return candidates


def hybrid_search(user_query: str,
                  knowledge_base_docs: list[dict],
                  semantic_retriever: DenseRetriever,
                  semantic_threshold: float = 0.5,
                  final_top_k: int = 5,
                  alpha: float = 0.7):
    """
    Performs a hybrid search: Semantic Search -> Field Re-ranking.
    """
    # 1. Parse User Prompt
    user_desires = parse_user_prompt(user_query)
    print(f"\n⚙️  Parsed Desires: {user_desires}")

    # 2. Initial Semantic Search to get a broad set of candidates
    print("\n--- (Stage 1) Performing Semantic Search ---")
    t0_sem = time.perf_counter()
    sem_docs, sem_scores = semantic_retriever.search(user_query, return_score=True)
    t1_sem = time.perf_counter()
    print(f"Found {len(sem_docs)} candidates in {1000 * (t1_sem - t0_sem):.1f} ms.")

    sem_score_map = {doc['id']: score for doc, score in zip(sem_docs, sem_scores)}
    doc_lookup_map = {doc['id']: doc for doc in knowledge_base_docs}

    # 3. Score and Filter Candidates
    print(f"\n--- (Stage 2) Filtering (Threshold > {semantic_threshold:.2f}) & Field Scoring ---")
    candidates = []
    for doc_id, sem_score in sem_score_map.items():
        if sem_score < semantic_threshold:
            continue

        doc = doc_lookup_map.get(doc_id)
        if doc:
            field_score = score_document(doc, user_desires)
            candidates.append({
                'doc': doc,
                'field_score': field_score,
                'semantic_score': sem_score,
            })

    # 4. Combine Scores and Re-rank
    print(f"\n--- (Stage 3) Combining Scores (alpha={alpha:.2f}) & Re-ranking ---")
    ranked_candidates = combine_scores(candidates, alpha)
    ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

    # 5. Return Final Top-K
    return ranked_candidates[:final_top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid Retriever for Squash Sessions")
    ap.add_argument("--query", required=True, help="User query for a squash session.")
    ap.add_argument("--sem_threshold", type=float, default=0.5, help="Minimum semantic score.")
    ap.add_argument("--final_topk", type=int, default=5, help="Final number of documents to return.")
    ap.add_argument("--retriever_cfg", default="configs/retrieval/faiss_rerank.yaml", help="Path to retriever YAML.")
    # NEW: Argument to control the weighting between field and semantic scores
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Weight for field score in final ranking (0.0 to 1.0). Default: 0.7")
    args = ap.parse_args()

    retriever_config_dict = load_cfg(args.retriever_cfg)
    KB_PATH = Path(retriever_config_dict['corpus_path'])

    if not KB_PATH.exists():
        print(f"❌ Error: Knowledge base not found at {KB_PATH}.")
        exit()

    with open(KB_PATH, "r", encoding="utf-8") as f:
        knowledge_base = [json.loads(line) for line in f if line.strip()]

    print(f"\n--- Initializing Semantic Retriever ---")
    semantic_retriever = DenseRetriever(retriever_config_dict)

    t0_total = time.perf_counter()
    results = hybrid_search(
        user_query=args.query,
        knowledge_base_docs=knowledge_base,
        semantic_retriever=semantic_retriever,
        semantic_threshold=args.sem_threshold,
        final_top_k=args.final_topk,
        alpha=args.alpha  # Pass the new alpha weight
    )
    t1_total = time.perf_counter()

    print(f"\n--- 🏆 Hybrid Retrieval Results (Total Time: {1000 * (t1_total - t0_total):.1f} ms) ---")
    if results:
        for i, res in enumerate(results):
            doc = res['doc']
            print(f"Rank {i + 1} (Final Score: {res['final_score']:.3f}): {doc.get('source')}")
            print(
                f"  Scores -> Field: {res['field_score']:.2f} (Normalized: {res.get('normalized_field_score', 0):.2f}), Semantic: {res['semantic_score']:.3f}")
            print(
                f"  Details -> Level: {doc.get('squashLevel')}, Duration: {doc.get('duration')}, Primary Shots: {doc.get('primaryShots')}")
            print("-" * 20)
    else:
        print("No documents found meeting the hybrid criteria.")