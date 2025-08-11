# rag/pipelines/retrieval/hybrid_retrieval/hybrid_retriever_rrf.py


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
from collections import defaultdict

# Monkey-patch torch.cuda if not available
if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self

from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever
from rag.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hybrid_search_rrf(user_query: str,
                      knowledge_base_docs: list[dict],
                      semantic_retriever: DenseRetriever,
                      final_top_k: int = 5,
                      rrf_k: int = 60):
    """
    Performs a hybrid search by running semantic and field retrievers in parallel
    and fusing their rankings using Reciprocal Rank Fusion (RRF).
    """
    # 1. Parse User Prompt
    user_desires = parse_user_prompt(user_query)
    print(f"\n⚙️  Parsed Desires: {user_desires}")
    doc_lookup_map = {doc['id']: doc for doc in knowledge_base_docs}

    # --- (Stage 1a) Run Semantic Retriever ---
    print("\n--- (Stage 1a) Performing Semantic Search ---")
    t0_sem = time.perf_counter()
    sem_docs, _ = semantic_retriever.search(user_query, return_score=True)
    t1_sem = time.perf_counter()
    print(f"Semantic search completed in {1000 * (t1_sem - t0_sem):.1f} ms.")
    # Create a map of {doc_id: rank}
    semantic_ranks = {doc['id']: i + 1 for i, doc in enumerate(sem_docs)}

    # --- (Stage 1b) Run Field Retriever ---
    print("\n--- (Stage 1b) Performing Field Scoring ---")
    t0_field = time.perf_counter()
    field_scored_docs = []
    for doc in knowledge_base_docs:
        field_score = score_document(doc, user_desires)
        if field_score > 0:
            field_scored_docs.append({'doc': doc, 'field_score': field_score})

    # Sort by field score to get the rank
    field_scored_docs.sort(key=lambda x: x['field_score'], reverse=True)
    t1_field = time.perf_counter()
    print(f"Field scoring completed in {1000 * (t1_field - t0_field):.1f} ms.")
    # Create a map of {doc_id: rank}
    field_ranks = {res['doc']['id']: i + 1 for i, res in enumerate(field_scored_docs)}

    # --- (Stage 2) Reciprocal Rank Fusion ---
    print(f"\n--- (Stage 2) Fusing ranks with RRF (k={rrf_k}) ---")
    rrf_scores = defaultdict(float)

    # Add scores from semantic ranking
    for doc_id, rank in semantic_ranks.items():
        rrf_scores[doc_id] += 1 / (rrf_k + rank)

    # Add scores from field ranking
    for doc_id, rank in field_ranks.items():
        rrf_scores[doc_id] += 1 / (rrf_k + rank)

    # --- (Stage 3) Get Final Ranked List ---
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)

    # Prepare final results with detailed info for analysis
    final_results = []
    field_score_map = {res['doc']['id']: res['field_score'] for res in field_scored_docs}

    for doc_id in sorted_doc_ids[:final_top_k]:
        final_results.append({
            'doc': doc_lookup_map[doc_id],
            'rrf_score': rrf_scores[doc_id],
            'semantic_rank': semantic_ranks.get(doc_id, 'N/A'),
            'field_rank': field_ranks.get(doc_id, 'N/A'),
            'field_score': field_score_map.get(doc_id, 0.0)
        })

    return final_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid Retriever using RRF for Squash Sessions")
    ap.add_argument("--query", required=True, help="User query for a squash session.")
    ap.add_argument("--final_topk", type=int, default=5, help="Final number of documents to return.")
    ap.add_argument("--retriever_cfg", default="rag/configs/retrieval/faiss_rerank.yaml",
                    help="Path to retriever YAML.")
    ap.add_argument("--k_rrf", type=int, default=60, help="The 'k' constant for Reciprocal Rank Fusion. Default: 60")
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
    results = hybrid_search_rrf(
        user_query=args.query,
        knowledge_base_docs=knowledge_base,
        semantic_retriever=semantic_retriever,
        final_top_k=args.final_topk,
        rrf_k=args.k_rrf
    )
    t1_total = time.perf_counter()

    print(f"\n--- 🏆 RRF Hybrid Retrieval Results (Total Time: {1000 * (t1_total - t0_total):.1f} ms) ---")
    if results:
        for i, res in enumerate(results):
            doc = res['doc']
            print(f"Rank {i + 1} (RRF Score: {res['rrf_score']:.4f}): {doc.get('source')}")
            print(
                f"  Ranks -> Semantic: {res['semantic_rank']}, Field: {res['field_rank']} (Score: {res['field_score']:.2f})")
            print(
                f"  Details -> Level: {doc.get('squashLevel')}, Duration: {doc.get('duration')}, Primary Shots: {doc.get('primaryShots')}")
            print("-" * 20)
    else:
        print("No documents found meeting the hybrid criteria.")