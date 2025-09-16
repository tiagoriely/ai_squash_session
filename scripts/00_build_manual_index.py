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
    # Try to import phrase helpers; fall back to no-op if unavailable
    try:
        from rag.utils import replace_phrases, SQUASH_PHRASES
    except Exception:
        def replace_phrases(s, _):  # no-op
            return s
        SQUASH_PHRASES = {}

    all_values = []
    for value in (meta or {}).values():
        if isinstance(value, list):
            list_str = " ".join([str(item) for item in value])
            list_str = replace_phrases(list_str, SQUASH_PHRASES)
            all_values.append(list_str)
        elif value is not None:
            value_str = replace_phrases(str(value), SQUASH_PHRASES)
            all_values.append(value_str)
    return " ".join(all_values)


def ensure_meta_and_contents(corpus: List[dict]) -> None:
    """
    Ensures each document has:
      - doc['meta'] as a dict (derived from all non-'contents' keys if missing)
      - doc['contents'] as a string (fallback from common alternatives, else empty)
    Modifies the corpus in place.
    """
    for doc in corpus:
        if not isinstance(doc, dict):
            continue

        # Ensure contents exists and is a string
        if 'contents' not in doc or doc['contents'] is None:
            for alt in ('content', 'text', 'body', 'document'):
                if alt in doc and doc[alt] is not None:
                    doc['contents'] = doc[alt]
                    break
            else:
                doc['contents'] = ""
        if not isinstance(doc['contents'], str):
            doc['contents'] = str(doc['contents'])

        # Ensure meta exists and is a dict
        if 'meta' not in doc or not isinstance(doc['meta'], dict):
            doc['meta'] = {k: v for k, v in doc.items() if k not in ('contents', 'meta')}


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

    # ✅ Normalise docs so downstream code can assume 'meta' and 'contents' exist
    ensure_meta_and_contents(corpus)

    # 1. --- Build Dense FAISS Index (Self-Contained) ---
    print("\n--- Building Dense (FAISS) Index ---")
    grammar_type = semantic_build_config['grammar_type']
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

        # Build texts for dense index: "passage: <meta>. <contents>"
        texts_for_dense_index = []
        for doc in tqdm(corpus, desc="Preparing Docs for Dense Index"):
            metadata_str = serialise_metadata(doc.get('meta', {}))
            contents_str = doc.get('contents', "")
            if not isinstance(contents_str, str):
                contents_str = str(contents_str)
            combined_text = f"passage: {metadata_str}. {contents_str}".strip()
            texts_for_dense_index.append(combined_text)

        all_embeddings = []
        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(texts_for_dense_index), batch_size), desc="Embedding Batches"):
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
    bm25_index_path = Path(sparse_build_config['index_path'])

    if bm25_index_path.exists() and not args.force:
        print(f"✅ BM25 index already exists at: {bm25_index_path}. Skipping.")
    else:
        bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        print("   - Serialising and tokenizing metadata for the corpus...")
        tokenized_corpus = [advanced_tokenizer(serialise_metadata(doc.get('meta', {}))) for doc in tqdm(corpus)]

        # Get k1 and b from the config, with sensible defaults
        k1 = sparse_build_config.get('k1', 1.2)
        b = sparse_build_config.get('b', 0.85)
        print(f"   - Initialising BM25 with k1={k1} and b={b}")

        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"✅ BM25 index built and saved to {bm25_index_path}")

    print("\n🎉 All indexes are built and ready for experiments!")
