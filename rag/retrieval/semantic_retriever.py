# rag/retrieval/semantic_retriever.py

from typing import List, Dict
import torch
import torch.nn as nn

# --- third-party ---
# Make sure flashrag is in your PYTHONPATH
from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever

# --- internal ---
from .base_retriever import BaseRetriever

# In case CUDA is not available, this prevents crashes when flashrag is imported.
if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self


class SemanticRetriever(BaseRetriever):
    """
    A wrapper for the FlashRAG DenseRetriever that conforms to the BaseRetriever interface.
    """

    def __init__(self, config: Dict):
        """
        Initializes the SemanticRetriever.

        Args:
            config (Dict): A configuration dictionary compatible with FlashRAG's DenseRetriever.
        """
        self._retriever = DenseRetriever(config)
        self._retrieval_top_k = config.get("retrieval_top_k", 100)  # Store original top_k

    def search(self, query: str, top_k: int) -> List[Dict]:
        """
        Performs a dense vector search using the FlashRAG retriever.

        Args:
            query (str): The user's search query.
            top_k (int): The maximum number of documents to return.

        Returns:
            List[Dict]: A ranked list of documents with their semantic scores.
        """
        # Temporarily set the top_k for this specific search call
        self._retriever.retrieval_top_k = top_k

        # query/passage optimisation technique
        prefixed_query = f"query: {query}"

        # Pass the prefixed query to the underlying retriever
        docs, scores = self._retriever.search(prefixed_query, return_score=True)
        # docs, scores = self._retriever.search(query, return_score=True)

        # Restore the original top_k from the config
        self._retriever.retrieval_top_k = self._retrieval_top_k

        # Combine documents and scores into a single list of dictionaries
        results = []
        for doc, score in zip(docs, scores):
            doc['semantic_score'] = float(score)  # Ensure score is a standard float
            results.append(doc)

        return results