# rag/retrieval/field_retriever.py

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any

from .base_retriever import BaseRetriever
# We now import from our new parsers package
from rag.parsers.user_query_parser import parse_user_prompt as _parse_user_prompt


class FieldRetriever(BaseRetriever):
    """
    A retriever that scores and ranks documents based on metadata field matching.
    """

    def __init__(self, knowledge_base: List[Dict], config_path: str | Path):
        """
        Initializes the FieldRetriever.

        Args:
            knowledge_base (List[Dict]): A list of document dictionaries to search through.
            config_path (str | Path): Path to the YAML configuration file.
        """
        self.kb = knowledge_base
        self._config = self._load_config_from_yaml(Path(config_path))
        self.synonym_map = self._config.get("SYNONYM_MAP", {})
        self.specific_maps = self._config.get("SPECIFIC_MAPS", {})
        self.field_scoring_config = self._config.get("FIELD_SCORING_CONFIG", {})

    def _load_config_from_yaml(self, path: Path) -> Dict:
        """Helper to load and parse the YAML config file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def search(self, query: str, top_k: int, allowed_durations: List[int] | None = None) -> List[Dict]:
        """
        Parses user desires, scores all documents in the KB, and returns the top_k.
        """
        user_desires = _parse_user_prompt(query, allowed_durations=allowed_durations)

        scored_documents = []
        for doc in self.kb:
            score = self._score_document(doc, user_desires)
            if score > 0:
                # Store both the doc and its score for ranking
                scored_documents.append({'doc': doc, 'field_score': score})

        # Sort documents by score in descending order
        scored_documents.sort(key=lambda x: x['field_score'], reverse=True)

        # Return just the document dictionaries, up to top_k
        return [item['doc'] for item in scored_documents[:top_k]]

    def _score_document(self, document: Dict, user_desires: Dict) -> float:
        """
        Scores a single document using field-specific helper functions based on configuration.
        This function acts as a dispatcher, calling the correct scoring logic for each field.
        """
        total_score = 0.0
        for field, user_val in user_desires.items():
            config = self.field_scoring_config.get(field)
            if not config:
                continue

            if user_val == 'no_preference':
                total_score += config.get("base_weight", 1.0) * 0.5
                continue

            method_name = config["method"]
            base_weight = config.get("base_weight", 1.0)

            # Use getattr to dynamically call the correct private scoring method
            scoring_method = getattr(self, method_name, None)

            if scoring_method:
                if method_name == "_score_exact_match_field":
                    total_score += scoring_method(user_val, document.get(field), base_weight)
                elif method_name == "_score_numerical_range_field":
                    total_score += scoring_method(user_val, document.get(field), base_weight, config.get("tolerance"))
                elif method_name == "_score_list_overlap_field":
                    total_score += scoring_method(user_val, document.get(field), base_weight, field)
                elif method_name == "_score_inferred_categorical_match_field":
                    total_score += scoring_method(user_val, document.get('intensity'), document.get('fitness'),
                                                  base_weight, config)
                elif method_name == "_score_hierarchical_boost_field":
                    total_score += scoring_method(user_val, document, base_weight, config)

        return total_score

    # --- All helper functions from field_matcher.py are now private methods ---

    def _clean_and_standardise_value(self, field: str, value_str: Any):
        if isinstance(value_str, list):
            return [item for item in (self._clean_and_standardise_value(field, v) for v in value_str) if
                    item is not None]

        raw_value_lower = str(value_str).lower().strip()

        if field in self.synonym_map:
            for synonym_key, canonical_form in self.synonym_map[field].items():
                if raw_value_lower == str(synonym_key).lower():
                    return canonical_form

        if field in ["participants", "duration"]:
            try:
                return int(re.search(r'\d+', raw_value_lower).group())
            except (ValueError, TypeError, AttributeError):
                pass

        return raw_value_lower

    def _calculate_proportional_overlap_score(self, user_set: set, doc_set: set, weight: float) -> float:
        if not user_set or not doc_set: return 0.0
        matched_items = user_set.intersection(doc_set)
        if not matched_items: return 0.0
        user_coverage = len(matched_items) / len(user_set)
        doc_focus = len(matched_items) / len(doc_set)
        combined_ratio = (0.7 * user_coverage) + (0.3 * doc_focus)
        return weight * combined_ratio

    def _score_exact_match_field(self, user_val, doc_val, base_weight):
        if user_val is None: return 0.0
        doc_val_set = set(doc_val if isinstance(doc_val, list) else [doc_val])
        user_val_set = set(user_val if isinstance(user_val, list) else [user_val])
        if user_val_set.intersection(doc_val_set):
            return base_weight
        return 0.0

    def _score_numerical_range_field(self, user_val, doc_val_raw, base_weight, tolerance=10):
        if user_val is None or doc_val_raw is None: return 0.0
        try:
            user_num = int(user_val)
            doc_num_match = re.search(r'\d+', str(doc_val_raw))
            if doc_num_match:
                doc_num = int(doc_num_match.group())
                deviation = abs(user_num - doc_num)
                if deviation <= tolerance:
                    return base_weight * (1 - (deviation / (tolerance + 1e-6)))
                elif deviation <= tolerance * 2:
                    return base_weight * 0.1
        except (ValueError, TypeError):
            pass
        return 0.0

    def _score_list_overlap_field(self, user_vals, doc_vals_raw, base_weight, field_name):
        if not user_vals or not doc_vals_raw: return 0.0
        doc_vals_list = doc_vals_raw if isinstance(doc_vals_raw, list) else [doc_vals_raw]
        doc_vals_standardised = set(self._clean_and_standardise_value(field_name, doc_vals_list))
        user_vals_set = set(user_vals)
        return self._calculate_proportional_overlap_score(user_vals_set, doc_vals_standardised, base_weight)

    def _score_inferred_categorical_match_field(self, user_val, doc_primary_val, doc_secondary_val, base_weight,
                                                config):
        if user_val is None: return 0.0
        if doc_primary_val == user_val: return base_weight
        if doc_secondary_val:
            context = config.get("field_name_for_secondary_standardisation")
            secondary_standardised = self._clean_and_standardise_value(context, doc_secondary_val)
            if secondary_standardised == user_val:
                return base_weight * config.get("secondary_match_multiplier", 0.75)
        return 0.0

    def _score_hierarchical_boost_field(self, user_vals: list, doc: dict, base_weight: float, config: dict):
        if not user_vals: return 0.0
        user_vals_set = set(user_vals)
        map_name = config["general_to_specific_map_name"]
        general_to_specific_map = self.specific_maps.get(map_name, {})

        user_general_shots = {s for s in user_vals_set if s in general_to_specific_map}
        user_specific_shots = user_vals_set - user_general_shots

        doc_main_field = set(self._clean_and_standardise_value("shots", doc.get("shots", [])))
        doc_primary_field = set(self._clean_and_standardise_value("shots", doc.get("primaryShots", [])))
        doc_secondary_field = set(self._clean_and_standardise_value("shots", doc.get("secondaryShots", [])))

        total_score = 0.0
        if user_general_shots and doc_main_field:
            total_score += self._calculate_proportional_overlap_score(user_general_shots, doc_main_field, base_weight)

        all_user_implied_shots = set(user_specific_shots)
        for general_shot in user_general_shots:
            all_user_implied_shots.update(general_to_specific_map.get(general_shot, []))

        if all_user_implied_shots:
            total_score += self._calculate_proportional_overlap_score(all_user_implied_shots, doc_primary_field,
                                                                      config["primary_boost_weight"])
            total_score += self._calculate_proportional_overlap_score(all_user_implied_shots, doc_secondary_field,
                                                                      config["secondary_boost_weight"])

        return total_score