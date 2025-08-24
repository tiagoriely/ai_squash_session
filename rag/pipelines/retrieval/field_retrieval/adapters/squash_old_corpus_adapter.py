# adapters/squash_old_corpus_adapter.py
from .base_adapter import BaseAdapter

class SquashOldCorpusAdapter(BaseAdapter):
    """Adapter for the original, flat-file squash corpus."""
    def transform(self, raw_doc: dict) -> dict:
        # The old corpus is already in the canonical format, so we just return it.
        # We could add cleaning/validation here if needed in the future.
        return raw_doc