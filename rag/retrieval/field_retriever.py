# In rag/retrieval/field_retriever.py

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any
from rapidfuzz import fuzz
from nltk import stem, PorterStemmer

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
        self.negative_keywords = self._config.get("NEGATIVE_KEYWORDS", {}).get("sports", [])
        self.stemmer = PorterStemmer()

    def _load_config_from_yaml(self, path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


    def search(self, query: str, top_k: int, allowed_durations: List[int] | None = None) -> List[Dict]:
        user_desires = _parse_user_prompt(query, allowed_durations=allowed_durations)
        scored_documents = []

        # Check for negative keywords in the query
        query_lower = query.lower()
        contains_squash = "squash" in query_lower

        if not contains_squash:
            for keyword in self.negative_keywords:
                if keyword in query_lower:
                    # The query is about another sport and doesn't mention squash.
                    # It's irrelevant, so we return no results.
                    return []  # Return empty list immediately

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

    def _score_fuzzy_match_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        """
        Scores a field based on fuzzy string similarity using rapidfuzz.
        Handles typos and minor variations.
        """
        doc_val = doc.get(field)
        if user_val is None or doc_val is None:
            return 0.0

        # Calculate the similarity ratio (0-100). The function call is identical to thefuzz.
        similarity_ratio = fuzz.ratio(str(user_val).lower(), str(doc_val).lower())

        # Get a similarity threshold from the config (e.g., 85%)
        threshold = config.get("similarity_threshold", 85)

        if similarity_ratio >= threshold:
            # Optional: Scale the score by how similar the match is
            score_multiplier = similarity_ratio / 100.0
            return config.get("base_weight", 1.0) * score_multiplier

        return 0.0

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

    def _normalise_session_type(self, value: str) -> str:
        """
        Normalises session type strings to handle common variations
        before synonym lookup.
        """
        if not isinstance(value, str):
            return value

        # 1. Lowercase and strip whitespace
        norm_value = value.lower().strip()

        # 2. Handle mix(...) patterns, e.g., "mix(cg, drill)" -> "mix"
        if norm_value.startswith("mix("):
            return "mix"

        # 3. Replace common separators with spaces, e.g., "conditioned_game" -> "conditioned game"
        norm_value = re.sub(r'[_,-]', ' ', norm_value)

        return norm_value

    # --- Scoring Methods ---

    def _score_exact_match_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        doc_val = doc.get(field)
        if user_val is None or doc_val is None:
            return 0.0

        # If we are scoring the 'type' field, normalize it first!
        if field == "type":
            doc_val = self._normalise_session_type(doc_val)

        # Standardize the document's value using the synonym map.
        doc_val_standardised = self._clean_and_standardise_value(field, doc_val)



        # Normalise both values to handle inconsistencies like underscores vs. spaces.
        norm_doc_val = str(doc_val_standardised).lower().replace('_', ' ')
        norm_user_val = str(user_val).lower().replace('_', ' ')

        # Compare the fully normalised values.
        if norm_doc_val == norm_user_val:
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

        # Standardise the document's list of values.
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

        # --- Part 1: Score General Overlap ---
        # This part is still valuable for broad queries like "a session with volleys".
        if user_general_shots and doc_main_field:
            total_score += self._calculate_proportional_overlap_score(user_general_shots, doc_main_field, base_weight)

        # --- Part 2: Score Specific Shot Boost ---
        # The boost is now ONLY calculated based on the specific shots the user
        # explicitly mentioned in their query (e.g., "volley cross").
        if user_specific_shots:
            # Check for matches in the document's primary shots
            total_score += self._calculate_proportional_overlap_score(
                user_specific_shots,
                doc_primary_field,
                config.get("primary_boost_weight", 6.0)
            )
            # Check for matches in the document's secondary shots
            total_score += self._calculate_proportional_overlap_score(
                user_specific_shots,
                doc_secondary_field,
                config.get("secondary_boost_weight", 4.0)
            )

        return total_score

    def _score_hierarchical_boost_field_with_hybrid_stem(self, doc: dict, field: str, user_vals: list, config: dict) -> float:
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

        # --- Part 1: Score General Overlap (STEMMING) ---
        if user_general_shots and doc_main_field:
            # Stem both sets before comparing for more flexible matching (e.g., 'drives' -> 'drive')
            user_general_stemmed = {self.stemmer.stem(s) for s in user_general_shots}
            doc_main_stemmed = {self.stemmer.stem(s) for s in doc_main_field}

            total_score += self._calculate_proportional_overlap_score(
                user_general_stemmed,
                doc_main_stemmed,
                base_weight
            )

        # --- Part 2: Score Specific Shot Boost ---
        # This part remains untouched to preserve the exact matching required for
        # specific multi-word phrases like "volley straight drive".
        if user_specific_shots:
            total_score += self._calculate_proportional_overlap_score(
                user_specific_shots,
                doc_primary_field,
                config.get("primary_boost_weight", 6.0)
            )
            total_score += self._calculate_proportional_overlap_score(
                user_specific_shots,
                doc_secondary_field,
                config.get("secondary_boost_weight", 4.0)
            )

        return total_score

    def _score_squash_level_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        """
        Custom scoring for squash level. Works with the flat data from the adapter.
        """
        if user_val is None:
            return 0.0

        # Check for an exact match in the primary 'squashLevel' field (recommended)
        if doc.get("squashLevel") == user_val:
            return config.get("base_weight", 1.0)

        # If no match, check if the user's desired level is in the applicable list
        applicable_levels = doc.get("applicable_squash_levels", [])
        if user_val in applicable_levels:
            return config.get("base_weight", 1.0) * 0.75

        return 0.0

    def _score_squash_level_hybrid_field(self, doc: dict, field: str, user_val: any, config: dict) -> float:
        """
        A hybrid scoring method for squash level that combines exact, list, and fuzzy matching.
        1. Checks for an exact match on the recommended level for the highest score.
        2. Checks if the level is in the 'applicable_squash_levels' list for a high score.
        3. As a fallback, uses fuzzy matching to catch typos for a slightly lower score.
        """
        if user_val is None:
            return 0.0

        base_weight = config.get("base_weight", 1.0)  # e.g., 4.0

        # --- Step 1: Exact Match (Same as your original logic) ---
        recommended_level = doc.get("squashLevel")
        if recommended_level and recommended_level == user_val:
            return base_weight

        # --- Step 2: Applicable List Match (Same as your original logic) ---
        applicable_levels = doc.get("applicable_squash_levels", [])
        if user_val in applicable_levels:
            return base_weight * 0.75  # Penalise slightly for not being the recommended level

        # --- Step 3: Fuzzy Match Fallback (The new addition) ---
        # Get fuzzy matching parameters from the config
        threshold = config.get("similarity_threshold", 85)
        fuzzy_penalty = config.get("fuzzy_penalty_multiplier", 0.6)  # Apply a penalty for fuzzy matches

        # Fuzzy check against the recommended level
        if recommended_level:
            similarity_ratio = fuzz.ratio(str(user_val).lower(), str(recommended_level).lower())
            if similarity_ratio >= threshold:
                return base_weight * fuzzy_penalty * (similarity_ratio / 100.0)

        # Fuzzy check against all applicable levels
        for level in applicable_levels:
            similarity_ratio = fuzz.ratio(str(user_val).lower(), str(level).lower())
            if similarity_ratio >= threshold:
                # Use the 0.75 multiplier for applicable levels, plus the fuzzy penalty
                return base_weight * 0.75 * fuzzy_penalty * (similarity_ratio / 100.0)

        return 0.0

    # --- Helper Methods ---

    def _clean_and_standardise_value(self, field: str, value: Any) -> Any:
        """
        Standardises a value from a document based on the rules in the config's SYNONYM_MAP.
        """
        if value is None:
            return None
        if isinstance(value, list):
            # Filter out None values that might result from failed standardisations in the list
            return [v for v in (self._clean_and_standardise_value(field, item) for item in value) if v is not None]

        # Use synonyms for standardisation
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

