# rag/retrieval/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    """
    Abstract Base Class for all retriever implementations.
    """
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
        """
        Searches the knowledge base for a given query.

        Args:
            query (str): The user's search query.
            top_k (int): The maximum number of documents to return.

        Returns:
            List[Dict]: A ranked list of documents, where each document is a dictionary.
        """
        pass