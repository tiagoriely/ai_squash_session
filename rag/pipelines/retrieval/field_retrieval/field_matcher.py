# rag/pipelines/retrieval/field_retrieval/field_matcher.py
# Reranker / ranking logic	-> Retriever · Correctness (evaluating ranking with labels)

import re
# Import the configuration from the new dedicated file
from rag.configs.field_retrieval_config import SYNONYM_MAP, FIELD_SCORING_CONFIG, SPECIFIC_MAPS


# MOVED TO CONFIG
# # -- 1. Synonym Mapping -- dictionary lookup
# # ... (your original commented-out block is preserved) ...

# --- Helper Functions (remain in field_matcher.py as they are core logic) ---

def clean_and_standardise_value(field: str, value_str: str | list):
    """
    Description: Cleans a string/list value and applies synonym mapping to get its canonical form.
    Handles lists by cleaning each item recursively.

    Parameters:
        - field (str): Field name used to pick the synonym map (e.g. "shots", "duration")
        - value_str (str | list): Raw 'value' or 'list of values' to normalise.

    Returns: Canonical string, integer (for numeric fields), or a list of canonical values.
    """
    if isinstance(value_str, list):
        # Filter out potential None values after cleaning
        return [item for item in (clean_and_standardise_value(field, v) for v in value_str) if item is not None]

    raw_value_lower = str(value_str).lower().strip()

    if field in SYNONYM_MAP:
        for synonym_key, canonical_form in SYNONYM_MAP[field].items():
            if raw_value_lower == str(synonym_key).lower():
                return canonical_form

    if field in ["participants", "duration"]:
        try:
            # Extract number from string like "60min"
            return int(re.search(r'\d+', raw_value_lower).group())
        except (ValueError, TypeError, AttributeError):
            pass

    return raw_value_lower


# --- START: CORRECTED PARSER LOGIC ---
def parse_user_prompt(prompt: str) -> dict:
    """
    Description: Parses a user prompt to extract desired field values, applying synonyms mapping.
    For 'shots', it extracts all detected canonical shot forms (both general and specific).

    Parameters:
        - prompt (str): The raw text of the user's query.

    Returns:
        dict: A dictionary of extracted and standardised user desires,
              with field names as keys and canonical values as values.
    """
    user_desires = {}
    prompt_lower = prompt.lower()

    # Helper to find all unique terms for a field using word boundaries for accuracy
    def _extract_terms(field_name):
        found_terms = set()
        # Sort terms by length, longest first, to match multi-word phrases correctly
        terms_to_check = sorted(SYNONYM_MAP.get(field_name, {}).keys(), key=len, reverse=True)
        for term in terms_to_check:
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', prompt_lower):
                found_terms.add(term)
        return found_terms


    # --- Extract single-value fields ---
    for field in ["type", "squashLevel", "intensity"]:
        terms = _extract_terms(field)
        if terms:
            # Take the first one found (usually the longest match)
            user_desires[field] = clean_and_standardise_value(field, list(terms)[0])

    # --- Extract numerical fields with specific regex ---
    participants_match = re.search(r'(\d+)\s*player|\b(one|two|three|four)\b|\bsolo\b', prompt_lower)
    if participants_match:
        val = participants_match.group(1) or participants_match.group(2) or "solo"
        user_desires["participants"] = clean_and_standardise_value("participants", val)

    duration_match = re.search(r'(\d+)\s*(?:min|hour)|an hour|hour and a half', prompt_lower)
    if duration_match:
        val = duration_match.group(1) or duration_match.group(0)
        user_desires["duration"] = clean_and_standardise_value("duration", val)

    # --- Extract multi-value fields ---
    side_terms = _extract_terms("shotSide")
    if side_terms:
        sides = {clean_and_standardise_value("shotSide", term) for term in side_terms}
        user_desires["shotSide"] = list(sides)

    movement_terms = _extract_terms("movement")
    if movement_terms:
        movements = {clean_and_standardise_value("movement", term) for term in movement_terms}
        user_desires["movement"] = list(movements)

    # --- Special, hierarchical extraction for shots ---
    all_shot_terms = set()
    # 1. Build a master list of all known shot phrases to search for
    general_shots_map = SPECIFIC_MAPS.get("GENERAL_SHOT_TYPES", {})
    all_possible_shots = set(SYNONYM_MAP.get("shots", {}).keys())
    for shots in general_shots_map.values():
        all_possible_shots.update(shots)

    # 2. Search for these phrases in the prompt (longest first for accuracy)
    for shot in sorted(all_possible_shots, key=len, reverse=True):
        if re.search(r'\b' + re.escape(shot.lower()) + r'\b', prompt_lower):
            # 3. If a shot is found, add the specific term AND its general canonical form
            all_shot_terms.add(shot)
            all_shot_terms.add(clean_and_standardise_value("shots", shot))

    if all_shot_terms:
        user_desires["shots"] = list(filter(None, all_shot_terms))

    # Final cleanup of any empty results
    return {k: v for k, v in user_desires.items() if v}


# --- END: CORRECTED PARSER LOGIC ---


# --- 2. Field-Specific Scoring Helper Functions ---

def _calculate_proportional_overlap_score(user_set: set, doc_set: set, weight: float) -> float:
    """
    Description: Calculates a score based on the overlap between user desired items and
    document items within a specific field, considering both how much of the user's
    request is covered and how focused the document's field is on those items.

    Parameters:
        - user_set (set): A set of canonical values desired by the user.
        - doc_set (set): A set of canonical values from the document's field.
        - weight (float): The base weight to multiply the combined ratio by.

    Returns:
        float: A score representing the proportional overlap, multiplied by the weight.
    """
    if not user_set or not doc_set:
        return 0.0
    matched_items = user_set.intersection(doc_set)
    if not matched_items:
        return 0.0
    user_coverage = len(matched_items) / len(user_set)
    doc_focus = len(matched_items) / len(doc_set)
    # Refined formula to prioritize user coverage
    combined_ratio = (0.7 * user_coverage) + (0.3 * doc_focus)
    return weight * combined_ratio


def _score_exact_match_field(user_val, doc_val, base_weight):
    """
    Description: Scores fields requiring an exact match between the user's desired value
    and the document's value. Typical for single-value categorical fields.

    Parameters:
        - user_val: The canonical value desired by the user for this field.
        - doc_val: The canonical value from the document for this field.
        - base_weight (float): The base weight assigned to this field in the scoring configuration.

    Returns:
        float: 'base_weight' if there's an exact match, otherwise 0.0.
    """
    if user_val is None:
        return 0.0
    # Refined logic to handle cases where doc_val or user_val can be lists
    doc_val_set = set(doc_val if isinstance(doc_val, list) else [doc_val])
    user_val_set = set(user_val if isinstance(user_val, list) else [user_val])
    if user_val_set.intersection(doc_val_set):
        return base_weight
    return 0.0


def _score_numerical_range_field(user_val, doc_val_raw, base_weight, tolerance=10):
    """
    Description: Scores numerical fields (like duration) based on proximity to the user's desired value.
    A score is awarded if the document's value falls within a specified tolerance,
    decreasing as the deviation from the user's value increases.

    Parameters:
        - user_val (int): The numerical value desired by the user (e.g., 45 for duration).
        - doc_val_raw (str | int): The raw duration value from the document (e.g., "40 minutes" or 40).
        - base_weight (float): The base weight for this field.
        - tolerance (int, optional): The maximum allowed difference for a full score. Defaults to 10.

    Returns:
        float: A score between 0.0 and `base_weight` depending on proximity.
    """
    if user_val is None or doc_val_raw is None:
        return 0.0
    try:
        user_num = int(user_val)
        doc_num_match = re.search(r'\d+', str(doc_val_raw))
        if doc_num_match:
            doc_num = int(doc_num_match.group())
            deviation = abs(user_num - doc_num)
            if deviation <= tolerance:
                return base_weight * (1 - (deviation / (tolerance + 1e-6)))  # Epsilon to avoid division by zero
            elif deviation <= tolerance * 2:
                return base_weight * 0.1  # Partial credit
    except (ValueError, TypeError):
        pass
    return 0.0


def _score_list_overlap_field(user_vals, doc_vals_raw, base_weight, field_name):
    """
    Description: Scores fields where both user desires and document values are lists,
    using a proportional overlap calculation.
    Typical for multi-value categorical fields like general 'shots' or 'movement'.

    Parameters:
        - user_vals (list): A list of canonical values desired by the user.
        - doc_vals_raw (list | str): The raw value(s) from the document.
        - base_weight (float): The base weight for this field.
        - field_name (str): The context (e.g. 'shots', 'movement') for standardization.

    Returns:
        float: A score reflecting the balance between user desire fulfilment and document field focus.
    """
    if not user_vals or not doc_vals_raw:
        return 0.0
    doc_vals_list = doc_vals_raw if isinstance(doc_vals_raw, list) else [doc_vals_raw]
    doc_vals_standardised = set(clean_and_standardise_value(field_name, doc_vals_list))
    user_vals_set = set(user_vals)
    return _calculate_proportional_overlap_score(user_vals_set, doc_vals_standardised, base_weight)


def _score_inferred_categorical_match_field(user_val, doc_primary_val, doc_secondary_val, base_weight, config):
    """
    Description: Scores a categorical field by checking for a direct match, with a fallback
    to an inferred match from a related secondary field (e.g., inferring 'intensity' from 'fitness').
    """
    if user_val is None: return 0.0
    if doc_primary_val == user_val: return base_weight
    if doc_secondary_val:
        context = config.get("field_name_for_secondary_standardisation")
        secondary_standardised = clean_and_standardise_value(context, doc_secondary_val)
        if secondary_standardised == user_val:
            return base_weight * config.get("secondary_match_multiplier", 0.75)
    return 0.0


def _score_hierarchical_boost_field(user_vals: list, doc: dict, base_weight: float, config: dict):
    """
    Description: Scores a field with a hierarchical boost system, differentiating between general mentions
    and specific primary/secondary focuses.
    """
    if not user_vals: return 0.0
    user_vals_set = set(user_vals)
    map_name = config["general_to_specific_map_name"]
    general_to_specific_map = SPECIFIC_MAPS.get(map_name, {})

    # Separate what the user asked for into general and specific terms
    user_general_shots = {s for s in user_vals_set if s in general_to_specific_map}
    user_specific_shots = user_vals_set - user_general_shots

    # Standardise the document's shot lists
    doc_main_field = set(clean_and_standardise_value("shots", doc.get("shots", [])))
    doc_primary_field = set(clean_and_standardise_value("shots", doc.get("primaryShots", [])))
    doc_secondary_field = set(clean_and_standardise_value("shots", doc.get("secondaryShots", [])))

    total_score = 0.0

    # 1. Base score for general category overlap
    if user_general_shots and doc_main_field:
        total_score += _calculate_proportional_overlap_score(user_general_shots, doc_main_field, base_weight)

    # Create a set of all specific shots the user is interested in (both explicit and implied)
    all_user_implied_shots = set(user_specific_shots)
    for general_shot in user_general_shots:
        all_user_implied_shots.update(general_to_specific_map.get(general_shot, []))

    # 2. Add boosts for matching specific shots in primary and secondary lists
    if all_user_implied_shots:
        total_score += _calculate_proportional_overlap_score(all_user_implied_shots, doc_primary_field,
                                                             config["primary_boost_weight"])
        total_score += _calculate_proportional_overlap_score(all_user_implied_shots, doc_secondary_field,
                                                             config["secondary_boost_weight"])

    return total_score


def score_document(document: dict, user_desires: dict) -> float:
    """
    Description: Scores a single document using field-specific helper functions based on configuration.
    This function acts as a dispatcher, calling the correct scoring logic for each field.
    """
    total_score = 0.0
    for field, user_val in user_desires.items():
        config = FIELD_SCORING_CONFIG.get(field)
        if not config: continue

        method_name = config["method"]
        base_weight = config.get("base_weight", 1.0)

        # Dispatch to the correct scoring function
        if method_name == "_score_exact_match_field":
            total_score += _score_exact_match_field(user_val, document.get(field), base_weight)
        elif method_name == "_score_numerical_range_field":
            total_score += _score_numerical_range_field(user_val, document.get(field), base_weight,
                                                        config.get("tolerance"))
        elif method_name == "_score_list_overlap_field":
            total_score += _score_list_overlap_field(user_val, document.get(field), base_weight, field)
        elif method_name == "_score_inferred_categorical_match_field":
            total_score += _score_inferred_categorical_match_field(user_val, document.get('intensity'),
                                                                   document.get('fitness'), base_weight, config)
        elif method_name == "_score_hierarchical_boost_field":
            total_score += _score_hierarchical_boost_field(user_val, document, base_weight, config)

    return total_score

# ... (your if __name__ == "__main__" block for testing) ...