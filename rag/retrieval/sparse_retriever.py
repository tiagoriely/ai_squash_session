# rag/retrieval/sparse_retriever.py

import pickle
from pathlib import Path
from typing import List, Dict

# Note the import from the renamed base_retriever.py
from .base_retriever import BaseRetriever


class SparseRetriever(BaseRetriever):
    """
    A retriever that uses a pre-built BM25 index for sparse lexical matching.
    """

    def __init__(self, knowledge_base: List[Dict], config: Dict):
        """
        Initializes the SparseRetriever by loading the BM25 index.

        Args:
            knowledge_base (List[Dict]): The full list of document dictionaries.
            config (Dict): The configuration dictionary, containing the index_path.
        """
        self.kb = knowledge_base
        self.config = config

        index_path = Path(self.config['index_path'])
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at: {index_path}. "
                "Please run scripts/01_build_indexes.py first."
            )

        print(f"   - Loading BM25 index from: {index_path}")
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, top_k: int) -> List[Dict]:
        """
        Performs a sparse search using the loaded BM25 index.

        Args:
            query (str): The user's search query.
            top_k (int): The maximum number of documents to return.

        Returns:
            List[Dict]: A ranked list of documents with their BM25 scores.
        """
        # 1. Tokenize the query in the same way the corpus was tokenized
        tokenized_query = query.lower().split(" ")

        # 2. Get the BM25 scores for all documents in the corpus
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 3. Combine scores with their original document indices
        indexed_scores = list(enumerate(doc_scores))

        # 4. Sort by score in descending order and get the top_k
        top_n_indices = sorted(indexed_scores, key=lambda item: item[1], reverse=True)[:top_k]

        # 5. Map the indices back to the full documents from the knowledge base
        results = []
        for index, score in top_n_indices:
            # Only return documents with a score greater than 0
            if score > 0:
                doc = self.kb[index].copy()  # Use .copy() to avoid modifying the original KB dict
                doc['sparse_score'] = score
                results.append(doc)

        return results