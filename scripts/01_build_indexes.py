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
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# --- Helper Function ---
def load_and_format_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    def _format_values(obj, context):
        if isinstance(obj, dict): return {k: _format_values(v, context) for k, v in obj.items()}
        if isinstance(obj, list): return [_format_values(elem, context) for elem in obj]
        if isinstance(obj, str):
            try:
                return obj.format(**context)
            except KeyError:
                return obj
        return obj

    return _format_values(config, config)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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
    faiss_index_path = Path(semantic_build_config[
                                'save_dir']) / f"{semantic_build_config['retrieval_method']}_{semantic_build_config['faiss_type']}.index"

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
        corpus_texts = ["query: " + doc['contents'] for doc in corpus]
        all_embeddings = []
        batch_size = 32  # You can adjust this based on your hardware
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus_texts), batch_size)):
                batch = corpus_texts[i:i + batch_size]
                inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
                outputs = model(**inputs)
                embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                # Normalize embeddings for E5 models
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
    # --- USE NEW CONFIG FOR PATHS ---
    bm25_index_path = Path(sparse_build_config['index_path'])
    corpus_path = Path(sparse_build_config['corpus_path'])

    if bm25_index_path.exists() and not args.force:
        print(f"✅ BM25 index already exists at: {bm25_index_path}. Skipping.")
    else:
        bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   - Loading corpus from: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]

        tokenized_corpus = [doc['contents'].split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"✅ BM25 index built and saved to {bm25_index_path}")

    print("\n🎉 All indexes are built and ready for experiments!")