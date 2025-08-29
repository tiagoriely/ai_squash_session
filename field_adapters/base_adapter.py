# adapters/base_adapter.py
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    """
    Abstract base class for a corpus adapter.
    Its job is to transform a raw document from a specific corpus
    into our standard, canonical format for scoring.
    """
    @abstractmethod
    def transform(self, raw_doc: dict) -> dict:
        """Takes a raw document and returns a canonical one."""
        pass