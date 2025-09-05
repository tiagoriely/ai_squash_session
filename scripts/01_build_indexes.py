# scripts/01_build_indexes.py
"""
Builds all necessary retrieval indexes for a given corpus using self-contained Python code.
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
from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from rank_bm25 import BM25Okapi

from rag.utils import load_and_format_config, advanced_tokenizer



# --- Helper Functions ---
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def serialise_metadata(meta: dict) -> str:
    """
    Converts a metadata dictionary into a single space-separated string
    with phrase-aware serialisation.
    """
    from rag.utils import replace_phrases, SQUASH_PHRASES  # Add imports

    all_values = []
    for value in meta.values():
        if isinstance(value, list):
            # Join list items with space, then apply phrase replacement
            list_str = " ".join([str(item) for item in value])
            list_str = replace_phrases(list_str, SQUASH_PHRASES)  # Apply phrase replacement
            all_values.append(list_str)
        elif value is not None:
            # Convert other values to string and apply phrase replacement
            value_str = str(value)
            value_str = replace_phrases(value_str, SQUASH_PHRASES)  # Apply phrase replacement
            all_values.append(value_str)
    return " ".join(all_values)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build all retrieval indexes.")
    parser.add_argument("--semantic-build-config", required=True,
                        help="Path to the semantic index builder YAML config.")
    parser.add_argument("--sparse-build-config", required=True, help="Path to the sparse retriever YAML config.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of indexes even if they exist.")
    args = parser.parse_args()

    print("▶️  Loading configurations...")
    semantic_build_config = load_and_format_config(args.semantic_build_config)
    sparse_build_config = load_and_format_config(args.sparse_build_config)

    corpus_path = Path(semantic_build_config['corpus_path'])
    print(f"   - Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]

    # 1. --- Build Dense FAISS Index (Self-Contained) ---
    print("\n--- Building Dense (FAISS) Index ---")
    # Get the corpus size from the config to use in the filename
    grammar_type = semantic_build_config['grammar_type']
    # Construct unique filename
    faiss_filename = f"{semantic_build_config['retrieval_method']}_{semantic_build_config['faiss_type']}_{grammar_type}.index"
    faiss_index_path = Path(semantic_build_config['save_dir']) / faiss_filename

    if faiss_index_path.exists() and not args.force:
        print(f"✅ FAISS index already exists at: {faiss_index_path}. Skipping.")
    else:
        faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"   - Loading model: {semantic_build_config['model_path']}...")
        tokenizer = AutoTokenizer.from_pretrained(semantic_build_config['model_path'])
        model = AutoModel.from_pretrained(semantic_build_config['model_path'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print(f"   - Encoding {len(corpus)} documents (using {device})...")
        corpus_texts = ["passage: " + doc['contents'] for doc in corpus]

        # Indexing metadata
        texts_for_dense_index = []
        for doc in tqdm(corpus, desc="Preparing Docs for Dense Index"):
            metadata_str = serialise_metadata(doc['meta'])
            combined_text = f"passage: {metadata_str}. {doc['contents']}"
            texts_for_dense_index.append(combined_text)

        all_embeddings = []
        batch_size = 32
        with torch.no_grad():
            # Use the new, explicitly named variable here
            for i in tqdm(range(0, len(texts_for_dense_index), batch_size)):
                batch = texts_for_dense_index[i:i + batch_size]
                inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
                outputs = model(**inputs)
                embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        embeddings_np = np.vstack(all_embeddings)

        print("   - Building FAISS index...")
        d = embeddings_np.shape[1]
        faiss_type = semantic_build_config.get('faiss_type', 'Flat')
        index = faiss.index_factory(d, faiss_type, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings_np)

        print(f"   - Saving FAISS index to: {faiss_index_path}")
        faiss.write_index(index, str(faiss_index_path))
        print(f"✅ FAISS index built with {index.ntotal} vectors.")

    # 2. --- Build Sparse BM25 Index (using Python) ---
    print("\n--- Building Sparse (BM25) Index ---")

    # --- CONFIG FOR PATHS ---
    bm25_index_path = Path(sparse_build_config['index_path'])

    if bm25_index_path.exists() and not args.force:
        print(f"✅ BM25 index already exists at: {bm25_index_path}. Skipping.")
    else:
        bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        print("   - Serialising and tokenizing metadata for the corpus...")
        tokenized_corpus = [advanced_tokenizer(serialise_metadata(doc['meta'])) for doc in tqdm(corpus)]

        # Get k1 and b from the config, with sensible defaults
        k1 = sparse_build_config.get('k1', 1.2)
        b = sparse_build_config.get('b', 0.85)
        print(f"   - Initialising BM25 with k1={k1} and b={b}")

        # Pass the parameters to the BM25 constructor
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"✅ BM25 index built and saved to {bm25_index_path}")

    print("\n🎉 All indexes are built and ready for experiments!")