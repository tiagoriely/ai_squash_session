import re
# Import the configuration from the new dedicated file
from rag.configs.field_retrieval_config import SYNONYM_MAP, FIELD_SCORING_CONFIG, SPECIFIC_MAPS

# MOVED TO CONFIG 👇🏼
# # -- 1. Synonym Mapping -- dictionary lookup
# # Maps user input terms to canonical field values
# # "word_in_prompt": "canonical_form"
# SYNONYM_MAP = {
#     "type": {
#         "drill": "drill", "routine": "drill",
#         "conditioned game": "conditioned game",
#         "solo practice": "solo", "solo": "solo", "by myself": "solo", "on my own": "solo", "alone on court": "solo", "no partner available": "solo",
#         "ghosting": "ghosting", "shadowing": "ghosting",
#     },
#     "participants": {
#         "1": 1, "one": 1, "solo": 1,
#         "2": 2, "two": 2, "duo": 2,
#         "3": 3, "three": 3, "trio": 3,
#         "4": 4, "four": 4,
#     },
#     "squashLevel": {
#         "beginner": "beginner", "novice": "beginner",
#         "intermediate": "intermediate",
#         "advanced": "advanced", "expert": "advanced", "pro": "advanced", "professional": "advanced",
#     },
#     "intensity": {
#         "low": "low",
#         "medium": "medium", "moderate": "medium",
#         "high": "high", "intense": "high", "hard": "high", "extremely high": "high"
#     },
#     "fitness": {
#         "low": "low",
#         "medium": "medium", "moderate": "medium",
#         "high": "high", "intense": "high", "hard": "high", "extremely high": "high"
#     },
#     # For `parse_user_prompt` to standardise, but `score_document` uses numerical
#     "duration": {
#         "10minutes": 10, "10 minutes": 10, "10min": 10, "10 min": 10,
#         "20minutes": 20, "20 minutes": 20, "20min": 20, "20 min": 20,
#         "30minutes": 30, "30 minutes": 30, "30min": 30, "30 min": 30,
#         "45minutes": 45, "45 minutes": 45, "45min": 45, "45 min": 45,
#         "60minutes": 60, "60 minutes": 60, "60min": 60, "60 min": 60,
#         "90minutes": 90, "90 minutes": 90, "90min": 90, "90 min": 90,
#     },
#     # Canonical types for general shot categories
#     "shots": {
#         "drive": "drive",
#         "cross": "cross",
#         "drop": "drop", "drops": "drop", "counter drop": "drop", "cross drop": "drop", "straight drop": "drop",
#         "boast": "boast", "boasts": "boast", "trickle boast": "boast", "reverse boast": "boast", "2-wall boast": "boast", "3-wall boast": "boast",
#         "lob": "lob", "lobs": "lob", "straight lob": "lob", "cross lob": "lob",
#         "volley": "volley", "volleys": "volley", "no bounce": "volley",
#         "kill": "kill", "straight kill": "kill", "cross kill": "kill",
#         "serve": "serve", "serves": "serve",
#         "nick": "nick", "nicks": "nick",
#         "flick": "flick", "flicks": "flick",
#     },
#     "shotSide": {
#         "forehand": "forehand", "fh": "forehand",
#         "backhand": "backhand", "bh": "backhand",
#         "both": "both", "either": "both", "forehand and backhand": "both", "backhand and forehand": "both", "fh and bh": "both", "bh and fh": "both",
#     },
#     "movement": {
#         "front": "front",
#         "middle": "middle",
#         "back": "back",
#         "sideways": "sideways", "lateral": "sideways", "side-to-side": "sideways",
#         "diagonal": "diagonal",
#         "multi-directional": "multi-directional", "all over the court": "multi-directional",
#     },
#
#     # --- ADD MORE HERE ---
# }
#
# # General shot types that might appear in other fields, in this case 'primaryShots' or 'secondaryShots' of retrieved Docs
# GENERAL_SHOT_TYPES = {
#     "drive": ["drive", "deep drive", "hard drive", "straight drive", "volley drive", "volley deep drive", "volley hard drive", "volley straight drive"],
#     "cross": ["cross", "cross-court", "cross court", "cross lob", "lob cross", "deep cross", "cross deep", "cross wide", "wide cross", "cross down the middle", "cross-court nick", "hard cross", "volley cross", "volley hard cross", "volley cross lob", "volley cross-court nick"],
#     "drop": ["drop", "counter drop", "cross drop", "straight drop", "volley cross drop", "volley straight drop"],
#     "boast": ["boast", "2-wall boast", "3-wall boast", "trickle boast", "reverse boast", "volley 2-wall boast", "volley 3-wall boast", "volley reverse boast"],
#     "lob": ["lob", "straight lob", "cross lob", "volley straight lob", "volley cross lob"],
#     "volley": ["volley", "volley drop", "volley drive", "volley cross", "volley lob", "volley flick", "volley 2-wall boast", "volley 3-wall boast", "volley reverse boast", "volley deep drive", "volley hard drive", "volley straight drive", "volley cross drop", "volley straight drop", "volley cross kill", "volley straight kill", "volley straight lob", "volley cross-court nick", "volley hard cross"],
#     "kill": ["straight kill", "volley straight kill", "cross kill", "volley cross kill"],
#     # "serve":
#     # "nick": ["cross-court nick", "volley cross-court nick"
#     # "flick":
# }


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
        return [clean_and_standardise_value(field, item) for item in value_str]

    raw_value_lower = str(value_str).lower().strip() # ensuring lower case and no trailing whitespace (spaces, tabs, newlines, etc..)

    if field in SYNONYM_MAP:
        # Iterate through the SYNONYM_MAP for the given field.
        # `synonym_key` is the KEY (e.g., "straight kill", "boasts")
        # `canonical_form` is the VALUE (e.g., "kill", "boast")
        for synonym_key, canonical_form in SYNONYM_MAP[field].items():
            # Check if the raw input value matches this synonym key from the map
            if raw_value_lower == str(synonym_key).lower():
                return canonical_form # Return the canonical form (the VALUE from the map)

    # If no synonym was found in the map for the given raw_value_lower,
    # then for numerical fields, try a direct integer conversion.
    if field in ["participants", "duration"]:
        try:
            return int(raw_value_lower)
        except ValueError:
            pass

    # If no synonym was found and it's not a numerical field, return the cleaned raw value itself.
    # This acts as a fallback for terms not in the map.
    return raw_value_lower


def parse_user_prompt(prompt: str) -> dict: # -> dict means 'expected return type is a dict'
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

    # Helper to extract and standardise based on a list of keywords/phrases
    def extract_and_standardise(field_name, terms_list, is_list=False):
        found_canonical_values = set()
        # Sort terms by length descending to match longer phrases first
        for term in sorted(terms_list, key=len, reverse=True):
            if term.lower() in prompt_lower:
                canonical = clean_and_standardise_value(field_name, term)
                if canonical is not None and canonical != []:
                    found_canonical_values.add(canonical)
        return list(found_canonical_values) if is_list else (list(found_canonical_values)[0] if found_canonical_values else None)

    # --- Type of session ---
    # Get all potential terms from the SYNONYM_MAP for 'type'
    type_terms = list(SYNONYM_MAP["type"].keys())
    user_desires["type"] = extract_and_standardise("type", type_terms)

    # --- Number of participants ---
    participants_match = re.search(r'(\d+)\s*player(?:s)?|\b(one|two|three|four)\s*players?\b|\bsolo\b', prompt_lower)
    if participants_match:
        if participants_match.group(1):  # Numeric match
            user_desires["participants"] = clean_and_standardise_value("participants", participants_match.group(1))
        elif participants_match.group(2):  # Word match (one, two, etc.)
            user_desires["participants"] = clean_and_standardise_value("participants", participants_match.group(2))
        elif "solo" in participants_match.group(0):  # Specifically handle "solo" if it's not caught by others
            user_desires["participants"] = clean_and_standardise_value("participants", "solo")

    # --- Squash Level ---
    level_terms = list(SYNONYM_MAP["squashLevel"].keys())
    user_desires["squashLevel"] = extract_and_standardise("squashLevel", level_terms)

    # --- Intensity ---
    intensity_terms = list(SYNONYM_MAP["intensity"].keys())
    user_desires["intensity"] = extract_and_standardise("intensity", intensity_terms)

    # --- Duration of session ---
    # Attempt to match specific phrases from SYNONYM_MAP first
    duration_terms = list(SYNONYM_MAP["duration"].keys())
    user_desires["duration"] = extract_and_standardise("duration", duration_terms)

    # If no specific phrase was matched, fall back to general numerical regex extraction
    if user_desires.get("duration") is None:
        duration_match = re.search(r'(\d+)\s*(?:min(?:ute)?s?)', prompt_lower)
        if duration_match:
            user_desires["duration"] = clean_and_standardise_value("duration", duration_match.group(1))

    # --- Shots (can be multiple) ---
    # This will now capture BOTH the general canonical form (e.g., 'kill') AND
    # the explicitly mentioned specific form (e.g., 'straight kill') if present in the prompt.
    all_detected_shot_terms = set()
    # Iterate through all possible shot terms (keys from SYNONYM_MAP for 'shots')
    # Process longer terms first to ensure specific matches are prioritized
    for term_in_map in sorted(SYNONYM_MAP["shots"].keys(), key=len, reverse=True):
        if term_in_map.lower() in prompt_lower:
            # Add the canonical form (e.g., "kill" for "straight kill" or "boast" for "boast")
            all_detected_shot_terms.add(clean_and_standardise_value("shots", term_in_map))

            # If the term itself is a specific shot and distinct from its canonical, add it too.
            # This relies on GENERAL_SHOT_TYPES containing these specific variations.
            if SPECIFIC_MAPS.get("GENERAL_SHOT_TYPES"):
                for specific_list in SPECIFIC_MAPS["GENERAL_SHOT_TYPES"].values():
                    if term_in_map in specific_list and term_in_map != clean_and_standardise_value("shots", term_in_map):
                        all_detected_shot_terms.add(term_in_map) # Add the explicit specific term itself
                        break # Found it, no need to check other specific lists

    if all_detected_shot_terms:
        user_desires["shots"] = list(all_detected_shot_terms)


    # --- Shot Side ---
    side_terms = list(SYNONYM_MAP["shotSide"].keys())
    user_desires["shotSide"] = extract_and_standardise("shotSide", side_terms)

    # --- Movement ---
    movement_terms = list(SYNONYM_MAP["movement"].keys())
    user_desires["movement"] = extract_and_standardise("movement", movement_terms, is_list=True)

    # --- ADD MORE HERE ---

    # Clean up None/empty list values
    user_desires = {k: v for k, v in user_desires.items() if v is not None and v != []}

    return user_desires

# --- 2. Field-Specific Scoring Helper Functions ---
# These functions are called dynamically by score_document based on FIELD_SCORING_CONFIG
# and remain in this file as they represent the scoring *logic*.
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
    matched_count = len(matched_items)

    if matched_count == 0:
        return 0.0

    user_coverage = matched_count / len(user_set)
    doc_focus = matched_count / len(doc_set) # This is the key part for your "proportion of matches based on doc size"

    combined_ratio = user_coverage * doc_focus

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
    if user_val is not None and doc_val == user_val:
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
        - tolerance (int, optional): The maximum allowed difference for a full score.
                                     Defaults to 10.

    Returns:
        float: A score between 0.0 and `base_weight` depending on proximity, or 0.0 if
               outside tolerance or input is invalid.
    """
    if user_val is None or doc_val_raw is None:
        return 0.0

    try:
        user_num = int(user_val)
        doc_num_match = re.search(r'\d+', str(doc_val_raw)) # Ensured doc_val_raw is string for regex
        if doc_num_match:
            doc_num = int(doc_num_match.group()) # returns that digit string (e.g., "45")
            deviation = abs(user_num - doc_num)
            if deviation <= tolerance:
                # Score decreases linearly with deviation, 0 at max tolerance
                return base_weight * (1 - (deviation / tolerance))
            elif deviation <= tolerance * 2:
                # Small partial credit for slightly outside
                PARTIAL_CREDIT = 0.1
                return base_weight * PARTIAL_CREDIT
    except ValueError:
        pass # Handle cases where conversion to int fail
    return 0.0

def _score_list_overlap_field(user_vals, doc_vals_raw, base_weight):
    """
    Description: Scores fields where both user desires and document values are lists,
    using a proportional overlap calculation.
    Typical for multi-value categorical fields like general 'shots' or 'movement'.

    Parameters:
        - user_vals (list): A list of canonical values desired by the user.
        - doc_vals_raw (list | str): The raw value(s) from the document, which could be a single string
                                     or a list of strings.
        - base_weight (float): The base weight for this field.

    Returns:
        float: A score reflecting the balance between user desire fulfilment and
               document field focus, multiplied by `base_weight`.
               Returns 0.0 if no user values or document values are provided.
    """
    if not user_vals or not doc_vals_raw:
        return 0.0

    # Ensure doc_vals are standardised sets for efficient intersection
    # Using "shots" as a placeholder field for clean_and_standardise_value context for list items.
    # This assumes doc_vals_raw's content is treated like "shots" terms for standardisation.
    doc_vals_standardised = set(
        clean_and_standardise_value("shots", doc_vals_raw) if isinstance(doc_vals_raw, list) else [
            clean_and_standardise_value("shots", doc_vals_raw)])
    user_vals_set = set(user_vals) if isinstance(user_vals, list) else {user_vals}

    # Now, just call the new helper for the actual scoring logic
    return _calculate_proportional_overlap_score(user_vals_set, doc_vals_standardised, base_weight)

def _score_inferred_categorical_match_field(user_val, doc_primary_val, doc_secondary_val,
                                            base_weight, secondary_match_multiplier=0.75,
                                            field_name_for_secondary_standardisation=None):
    """
    Description: Scores a categorical field by checking for a direct match with a primary document value.
    If no direct match, it attempts to infer a match from a related, secondary document value
    by first standardising the secondary value. This is useful when one field's value
    can imply another (e.g., fitness implying intensity).

    Currently used for:
        - 'intensity' field, where 'doc_primary_val' is 'intensity' and 'doc_secondary_val' is 'fitness'.
          In this specific use case, 'field_name_for_secondary_standardisation' would be 'intensity'.

    Parameters:
        - user_val: The canonical categorical value desired by the user.
        - doc_primary_val: The canonical value from the document's primary field (e.g., 'intensity').
        - doc_secondary_val: The raw value from the document's secondary, related field (e.g., 'fitness').
        - base_weight (float): The base weight for this field.
        - secondary_match_multiplier (float, optional): A multiplier applied to the base_weight
                                                     if the match is inferred from the secondary value.
                                                     Defaults to 0.75.
        - field_name_for_secondary_standardisation (str, optional): The field name (key in SYNONYM_MAP)
                                                                 to use when standardising `doc_secondary_val`.
                                                                 This is crucial if `doc_secondary_val`'s raw
                                                                 content maps to a canonical value of a different field.
                                                                 If None, it defaults to the primary field's implicit context.

    Returns:
        float: The calculated score based on direct or inferred match. Returns 0.0 if user_val is None.
    """
    current_score = 0.0
    if user_val is None:
        return 0.0

    if doc_primary_val == user_val:
        current_score += base_weight
    elif doc_secondary_val is not None:
        standardisation_context_field = field_name_for_secondary_standardisation if field_name_for_secondary_standardisation else "intensity"

        doc_secondary_val_standardised = clean_and_standardise_value(
            standardisation_context_field,
            doc_secondary_val
        )
        if doc_secondary_val_standardised == user_val:
            current_score += base_weight * secondary_match_multiplier
    return current_score

def _score_hierarchical_boost_field(
    user_desired_vals: list,             # user_desires['shots'] (e.g., ['straight kill', 'kill'])
    doc_main_field_raw: list | str,      # document['shots']
    doc_primary_boost_field_raw: list | str,
    doc_secondary_boost_field_raw: list | str,
    base_weight: float,
    primary_boost_weight: float,
    secondary_boost_weight: float,
    field_name_for_standardisation: str,
    general_to_specific_map: dict | None = None # GENERAL_SHOT_TYPES
):
    """
    Description: Scores a field with a hierarchical boost system, differentiating between general mentions
    and specific primary/secondary focuses. It awards base points for general matches and additional
    points for specific matches, with a special bonus for documents whose primary focus is exclusively
    aligned with the user's specific desires.

    This function expects:
    - `user_desired_vals` to contain *both* general and specific canonical forms (e.g., ['boast', '3-wall boast'] if user mentioned '3-wall boast').
    - `doc_main_field_raw` to contain general categories from the document.
    - `doc_primary_boost_field_raw` and `doc_secondary_boost_field_raw` to contain specific variations.
    - `general_to_specific_map` to map general categories to their specific variations for cross-checking.

    Currently used for: 'shots' field.

    Parameters:
        - user_desired_vals (list): A list of canonical general and specific values desired by the user.
        - doc_main_field_raw (list | str): The raw value(s) from the document's main field (general categories).
        - doc_primary_boost_field_raw (list | str): Raw primary focus values from document (specific variations).
        - doc_secondary_boost_field_raw (list | str): Raw secondary focus values from document (specific variations).
        - base_weight (float): Base weight for general matches.
        - primary_boost_weight (float): Additional score for primary focus matches.
        - secondary_boost_weight (float): Additional score for secondary focus matches.
        - field_name_for_standardisation (str): Field name for `clean_and_standardise_value`.
        - general_to_specific_map (dict, optional): Map from general to specific variations.

    Returns:
        float: Total score for the field.
    """
    current_score = 0.0
    if not user_desired_vals:
        return 0.0

    user_desired_vals_set = set(user_desired_vals) # e.g., {'straight kill', 'kill'}

    # Standardise all relevant document fields to canonical forms
    doc_main_field_standardised = set(clean_and_standardise_value(field_name_for_standardisation, doc_main_field_raw if isinstance(doc_main_field_raw, list) else [doc_main_field_raw])) if doc_main_field_raw else set()
    doc_primary_boost_field_standardised = set(clean_and_standardise_value(field_name_for_standardisation, doc_primary_boost_field_raw if isinstance(doc_primary_boost_field_raw, list) else [doc_primary_boost_field_raw])) if doc_primary_boost_field_raw else set()
    doc_secondary_boost_field_standardised = set(clean_and_standardise_value(field_name_for_standardisation, doc_secondary_boost_field_raw if isinstance(doc_secondary_boost_field_raw, list) else [doc_secondary_boost_field_raw])) if doc_secondary_boost_field_raw else set()

    # Determine user's *implied* specific shots (from general terms and explicit specific terms)
    user_implied_specific_shots = set()
    for user_val in user_desired_vals_set:
        if general_to_specific_map and user_val in general_to_specific_map: # If it's a general category (like "kill")
            user_implied_specific_shots.update(general_to_specific_map[user_val])
        else: # It's a specific term (like "straight kill") or not in map, add directly
            user_implied_specific_shots.add(user_val)

    # 1. Base score for matching in the main field (doc['shots'] - general categories)
    # Use proportional overlap for general categories
    current_score += _calculate_proportional_overlap_score(user_desired_vals_set.intersection(doc_main_field_standardised), doc_main_field_standardised, base_weight)


    # 2. Proportional boost for matches in primary boost field
    # We want to score how well the primary shots align with *all* user desires (general or specific)
    # that *could* be specific shots. So, we use user_implied_specific_shots for this.
    current_score += _calculate_proportional_overlap_score(user_implied_specific_shots, doc_primary_boost_field_standardised, primary_boost_weight)

    # 3. Proportional boost for matches in secondary boost field
    # Similarly, for secondary shots.
    current_score += _calculate_proportional_overlap_score(user_implied_specific_shots, doc_secondary_boost_field_standardised, secondary_boost_weight)


    # 4. "ONLY" bonus for perfect primary focus
    # This bonus applies if the document's primary specific shots are a perfect subset of what the user implied/desired,
    # and there's actually some overlap (not empty sets).
    PERFECT_FOCUS_BONUS = primary_boost_weight * 1.5 # Extra bonus for perfect primary focus

    if doc_primary_boost_field_standardised and \
       doc_primary_boost_field_standardised.issubset(user_implied_specific_shots) and \
       user_implied_specific_shots.intersection(doc_primary_boost_field_standardised):
        # Additional condition for very high alignment: if the primary field has only what the user wants (or subset)
        # We can add a factor based on how many *more* things it has.
        # Let's say if the primary field has exactly what the user wants, or just one extra.
        # This condition means all documented primary shots are relevant to the user's specific desires.
        current_score += PERFECT_FOCUS_BONUS


    return current_score

def score_document(document: dict, user_desires: dict) -> float:
    """
    Description: Scores a single document using field-specific helper functions based on configuration.

    Parameters:
        - document (dict): The document from the knowledge base to be scored.
        - user_desires (dict): A dictionary of extracted and standardised user desires.

    Returns:
        float: The total calculated score for the document.
    """
    total_score = 0.0

    for field, user_val in user_desires.items():
        config = FIELD_SCORING_CONFIG.get(field)
        if not config:
            # If a field from user_desires is not in the scoring config, it means
            # we haven't defined how to score it, so it contributes 0.
            continue

        scoring_method_name = config["method"]
        base_weight = config.get("base_weight", 0.0) # Use .get with default for base_weight

        general_to_specific_map = None
        if "general_to_specific_map_name" in config:
            map_name = config["general_to_specific_map_name"]
            general_to_specific_map = SPECIFIC_MAPS.get(map_name)

        if scoring_method_name == "_score_exact_match_field":
            total_score += _score_exact_match_field(user_val, document.get(field), base_weight)
        elif scoring_method_name == "_score_numerical_range_field":
            total_score += _score_numerical_range_field(user_val, document.get(field), base_weight, config.get("tolerance"))
        elif scoring_method_name == "_score_list_overlap_field":
            total_score += _score_list_overlap_field(user_val, document.get(field), base_weight)
        elif scoring_method_name == "_score_inferred_categorical_match_field":
            total_score += _score_inferred_categorical_match_field(
                user_val,
                document.get('intensity'), # doc_primary_val for 'intensity'
                document.get('fitness'),   # doc_secondary_val for 'intensity'
                base_weight,
                config.get("secondary_match_multiplier"),
                config.get("field_name_for_secondary_standardisation")
            )
        elif scoring_method_name == "_score_hierarchical_boost_field":
            # This is the correct call for the generic hierarchical function
            total_score += _score_hierarchical_boost_field(
                user_val, # user_desired_vals (e.g., user_desires['shots'] - now contains both general and specific)
                document.get('shots'), # doc_main_field_raw
                document.get('primaryShots'), # doc_primary_boost_field_raw
                document.get('secondaryShots'), # doc_secondary_boost_field_raw
                base_weight,
                config["primary_boost_weight"],
                config["secondary_boost_weight"],
                config["field_name_for_standardisation"], # "shots"
                general_to_specific_map # Pass the dynamically retrieved map
            )
        # There should be no _score_explicit_specific_shots_field here, as its logic is now in _score_hierarchical_boost_field
        # Add more `elif` conditions for other custom scoring methods
    return total_score