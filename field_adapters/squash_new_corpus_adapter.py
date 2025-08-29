# adapters/squash_new_corpus_adapter.py
from .base_adapter import BaseAdapter

class SquashNewCorpusAdapter(BaseAdapter):
    """Adapter for the new, nested 'balanced_grammar' squash corpus."""
    def transform(self, raw_doc: dict) -> dict:
        meta = raw_doc.get("meta", {})
        return {
            "id": raw_doc.get("id"),
            "source": meta.get("archetype"),
            "type": meta.get("session_type"),
            "participants": meta.get("participants"),
            "duration": meta.get("duration"),
            "squashLevel": meta.get("recommended_squash_level"),
            "intensity": meta.get("intensity"), # This key may not exist in the new corpus
            "primaryShots": meta.get("shots_specific_primary", []),
            "secondaryShots": meta.get("shots_specific_secondary", []),
            "shots": meta.get("shots_general", []),
            "shotSide": meta.get("shotSide", []),
            # You can add any other fields you need for display/evaluation
            "contents": raw_doc.get("contents"),
        }