
# python3 -m rag.pipelines.retrieval.hybrid_retrieval.hybrid_retriever_field_then_semantic \
# --query "I want an advanced drill for 2 players focusing on cross lobs with medium intensity lasting about 45 minutes." \
# --field_threshold 1.0 \
# --final_topk 5 \
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

if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self

from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever
from rag.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document

def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def hybrid_search_field_first(user_query: str,
                              knowledge_base_docs: list[dict],
                              semantic_retriever: DenseRetriever,
                              field_threshold: float = 1.0,
                              final_top_k: int = 5):
    """
    Performs a hybrid search: Field Filter -> Semantic Search.
    """
    # 1. Parse User Prompt
    user_desires = parse_user_prompt(user_query)
    print(f"\n⚙️  Parsed Desires: {user_desires}")

    # 2. Score ALL documents in the KB with the field matcher
    print(f"\n--- (Stage 1) Scoring all {len(knowledge_base_docs)} documents with Field Matcher ---")
    t0_field = time.perf_counter()
    field_candidates = []
    for doc in knowledge_base_docs:
        field_score = score_document(doc, user_desires)
        if field_score >= field_threshold:
            field_candidates.append({'doc': doc, 'field_score': field_score})
    t1_field = time.perf_counter()
    print(f"Found {len(field_candidates)} candidates above threshold {field_threshold:.2f} in {1000 * (t1_field - t0_field):.1f} ms.")

    if not field_candidates:
        return []

    # 3. Use Semantic Search to get scores for the filtered candidates
    # NOTE: This is a simulation. A true semantic re-ranker would be different.
    # Here, we get scores for ALL docs and then pluck the ones we need.
    print("\n--- (Stage 2) Getting Semantic Scores for Candidates ---")
    sem_docs, sem_scores = semantic_retriever.search(user_query, return_score=True)
    sem_score_map = {doc['id']: score for doc, score in zip(sem_docs, sem_scores)}

    # Add semantic scores to our field candidates
    for candidate in field_candidates:
        doc_id = candidate['doc'].get('id')
        candidate['semantic_score'] = sem_score_map.get(doc_id, 0.0) # Default to 0 if not in top semantic results

    # 4. Rank candidates by Field Score first, then Semantic Score as a tie-breaker
    field_candidates.sort(key=lambda x: (x['field_score'], x['semantic_score']), reverse=True)

    return field_candidates[:final_top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid Retriever (Field-First) for Squash Sessions")
    ap.add_argument("--query", required=True, help="User query for a squash session.")
    ap.add_argument("--field_threshold", type=float, default=1.0, help="Minimum field score to consider a document. Default: 1.0")
    ap.add_argument("--final_topk", type=int, default=5, help="Final number of documents to return.")
    ap.add_argument("--retriever_cfg", default="configs/retrieval/faiss_rerank.yaml", help="Path to retriever YAML.")
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
    results = hybrid_search_field_first(
        user_query=args.query,
        knowledge_base_docs=knowledge_base,
        semantic_retriever=semantic_retriever,
        field_threshold=args.field_threshold,
        final_top_k=args.final_topk
    )
    t1_total = time.perf_counter()

    print(f"\n--- 🏆 Hybrid Retrieval Results (Total Time: {1000 * (t1_total - t0_total):.1f} ms) ---")
    if results:
        for i, res in enumerate(results):
            doc = res['doc']
            print(f"Rank {i + 1} (Field Score: {res['field_score']:.2f}): {doc.get('source')}")
            print(f"  Scores -> Semantic: {res['semantic_score']:.3f}")
            print(f"  Details -> Level: {doc.get('squashLevel')}, Duration: {doc.get('duration')}, Primary Shots: {doc.get('primaryShots')}")
            print("-" * 20)
    else:
        print("No documents found meeting the hybrid criteria.")