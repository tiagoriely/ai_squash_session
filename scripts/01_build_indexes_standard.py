# scripts/01_build_indexes_standard.py

"""
Builds retrieval indexes for a "standard sparse" setup, where the BM25
index is built on the full document 'contents' field.
"""
import argparse
import json
import yaml
import pickle
from pathlib import Path
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from rag.utils import load_and_format_config, advanced_tokenizer
from rank_bm25 import BM25Okapi


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retrieval indexes for a standard (full-text) sparse setup.")
    parser.add_argument("--semantic-build-config", required=True,
                        help="Path to the semantic index builder YAML config.")
    parser.add_argument("--sparse-build-config", required=True,
                        help="Path to the standard sparse retriever YAML config.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of indexes even if they exist.")
    args = parser.parse_args()

    print("▶️  Loading configurations...")
    semantic_build_config = load_and_format_config(args.semantic_build_config)
    sparse_build_config = load_and_format_config(args.sparse_build_config)

    corpus_path = Path(semantic_build_config['corpus_path'])
    print(f"   - Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]

    # --- Build Dense FAISS Index (This part remains the same) ---
    # ... (The FAISS index logic from your original script can be copied here as it already uses 'contents')

    # --- Build Standard Sparse BM25 Index ---
    print("\n--- Building Standard Sparse (BM25) Index on 'contents' ---")
    bm25_index_path = Path(sparse_build_config['index_path'])

    if bm25_index_path.exists() and not args.force:
        print(f"✅ BM25 index already exists at: {bm25_index_path}. Skipping.")
    else:
        bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

        # Standard Sparse (not Meta)
        # We now tokenize the 'contents' field instead of the serialised metadata.
        print("   - Tokenizing the 'contents' field for the corpus...")
        tokenized_corpus = [advanced_tokenizer(doc['contents']) for doc in tqdm(corpus)]

        k1 = sparse_build_config.get('k1', 1.2)
        b = sparse_build_config.get('b', 0.85)
        print(f"   - Initialising BM25 with k1={k1} and b={b}")

        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"✅ Standard BM25 index built and saved to {bm25_index_path}")

    print("\n🎉 All standard indexes are built.")