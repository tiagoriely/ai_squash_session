# In rag/retrieval/field_retriever.py

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any

from .base_retriever import BaseRetriever
from rag.parsers.user_query_parser import parse_user_prompt as _parse_user_prompt


class FieldRetriever(BaseRetriever):
    """
    A retriever that scores and ranks documents based on metadata field matching.
    """

    def __init__(self, knowledge_base: List[Dict], config_path: str | Path):
        self.kb = knowledge_base
        self.config_path = Path(config_path)
        self._config = self._load_config_from_yaml(self.config_path)
        self.synonym_map = self._config.get("SYNONYM_MAP", {})
        self.specific_maps = self._config.get("SPECIFIC_MAPS", {})
        self.field_scoring_config = self._config.get("FIELD_SCORING_CONFIG", {})

    def _load_config_from_yaml(self, path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def search(self, query: str, top_k: int, allowed_durations: List[int] | None = None) -> List[Dict]:
        user_desires = _parse_user_prompt(query, allowed_durations=allowed_durations)
        scored_documents = []
        for doc in self.kb:
            score = self._score_document(doc, user_desires)
            if score > 0:
                scored_documents.append({'doc': doc, 'field_score': score})

        scored_documents.sort(key=lambda x: x['field_score'], reverse=True)
        results = []
        for item in scored_documents[:top_k]:
            doc = item['doc'].copy()
            doc['field_score'] = item['field_score']
            results.append(doc)
        return results

    def _score_document(self, document: Dict, user_desires: Dict) -> float:
        """
        Scores a single document by dispatching to the correct scoring logic for each field.
        """
        total_score = 0.0
        for field, user_val in user_desires.items():
            config = self.field_scoring_config.get(field)
            if not config:
                continue

            method_name = config.get("method")
            if not method_name:
                continue

            scoring_method = getattr(self, method_name, None)
            if scoring_method:
                # Call the specific scoring method with the document, field name, user value, and config
                total_score += scoring_method(document, field, user_val, config)
        return total_score

    # --- Scoring Methods ---

    def _score_exact_match_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        doc_val = doc.get(field)
        if user_val is None or doc_val is None:
            return 0.0

        # Standardize the document's value before comparing.
        doc_val_standardised = self._clean_and_standardise_value(field, doc_val)

        # Compare the standardized document value with the already-standardized user value.
        if str(doc_val_standardised) == str(user_val):
            return config.get("base_weight", 1.0)
        return 0.0

    def _score_numerical_range_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        doc_val_raw = doc.get(field)
        if user_val is None or doc_val_raw is None:
            return 0.0

        tolerance = config.get("tolerance", 10)
        try:
            user_num = int(user_val)
            doc_num = int(doc_val_raw)
            deviation = abs(user_num - doc_num)
            if deviation <= tolerance:
                return config.get("base_weight", 1.0) * (1 - (deviation / (tolerance + 1e-6)))
        except (ValueError, TypeError):
            pass
        return 0.0

    def _score_list_overlap_field(self, doc: dict, field: str, user_vals: list, config: dict) -> float:
        doc_vals_raw = doc.get(field)
        if not user_vals or not doc_vals_raw:
            return 0.0

        doc_vals_list = doc_vals_raw if isinstance(doc_vals_raw, list) else [doc_vals_raw]

        # Standardize the document's list of values.
        doc_vals_standardised = set(self._clean_and_standardise_value(field, doc_vals_list))
        user_vals_set = set(user_vals)

        return self._calculate_proportional_overlap_score(user_vals_set, doc_vals_standardised,
                                                          config.get("base_weight", 1.0))

    def _score_inferred_categorical_match_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        if user_val is None:
            return 0.0

        base_weight = config.get("base_weight", 1.0)
        doc_primary_val = doc.get('intensity')  # Example primary field
        doc_secondary_val = doc.get('fitness')  # Example secondary field

        if doc_primary_val == user_val:
            return base_weight

        if doc_secondary_val:
            context = config.get("field_name_for_secondary_standardisation")
            secondary_standardised = self._clean_and_standardise_value(context, doc_secondary_val)
            if secondary_standardised == user_val:
                return base_weight * config.get("secondary_match_multiplier", 0.75)
        return 0.0

    def _score_hierarchical_boost_field(self, doc: dict, field: str, user_vals: list, config: dict) -> float:
        if not user_vals:
            return 0.0

        user_vals_set = set(user_vals)
        map_name = config["general_to_specific_map_name"]
        general_to_specific_map = self.specific_maps.get(map_name, {})

        user_general_shots = {s for s in user_vals_set if s in general_to_specific_map}
        user_specific_shots = user_vals_set - user_general_shots

        doc_main_field = set(self._clean_and_standardise_value("shots", doc.get("shots", [])))
        doc_primary_field = set(self._clean_and_standardise_value("shots", doc.get("primaryShots", [])))
        doc_secondary_field = set(self._clean_and_standardise_value("shots", doc.get("secondaryShots", [])))

        total_score = 0.0
        base_weight = config.get("base_weight", 1.0)
        if user_general_shots and doc_main_field:
            total_score += self._calculate_proportional_overlap_score(user_general_shots, doc_main_field, base_weight)

        all_user_implied_shots = set(user_specific_shots)
        for general_shot in user_general_shots:
            all_user_implied_shots.update(general_to_specific_map.get(general_shot, []))

        if all_user_implied_shots:
            total_score += self._calculate_proportional_overlap_score(all_user_implied_shots, doc_primary_field,
                                                                      config.get("primary_boost_weight", 1.5))
            total_score += self._calculate_proportional_overlap_score(all_user_implied_shots, doc_secondary_field,
                                                                      config.get("secondary_boost_weight", 1.2))
        return total_score

    # --- Helper Methods ---

    def _clean_and_standardise_value(self, field: str, value: Any) -> Any:
        """
        Standardizes a value from a document based on the rules in the config's SYNONYM_MAP.
        """
        if value is None:
            return None
        if isinstance(value, list):
            # Filter out None values that might result from failed standardizations in the list
            return [v for v in (self._clean_and_standardise_value(field, item) for item in value) if v is not None]

        # Use synonyms for standardization
        if field in self.synonym_map:
            for syn, canon in self.synonym_map[field].items():
                if str(value).lower().strip() == str(syn).lower().strip():
                    return canon

        return value

    def _calculate_proportional_overlap_score(self, user_set: set, doc_set: set, weight: float) -> float:
        if not user_set or not doc_set:
            return 0.0
        matched_items = user_set.intersection(doc_set)
        if not matched_items:
            return 0.0
        user_coverage = len(matched_items) / len(user_set)
        doc_focus = len(matched_items) / len(doc_set)
        combined_ratio = (0.7 * user_coverage) + (0.3 * doc_focus)
        return weight * combined_ratio