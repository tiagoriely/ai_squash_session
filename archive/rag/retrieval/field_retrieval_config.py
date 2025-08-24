# rag/config/field_retrieval_config.py

# --- Synonym Mapping ---
# Maps user input terms to canonical field values.
# This dictionary will be imported by field_matcher.py
SYNONYM_MAP = {
    "type": {
        "drill": "drill", "routine": "drill",
        "conditioned game": "conditioned game", "game": "conditioned game",
        "solo practice": "solo practice", "solo": "solo practice", "by myself": "solo practice", "on my own": "solo practice", "alone on court": "solo practice", "no partner available": "solo practice",
        "ghosting": "ghosting", "shadowing": "ghosting",
    },
    "participants": {
        "1": 1, "one": 1, "solo": 1,
        "2": 2, "two": 2, "duo": 2, "second player": 2,
        "3": 3, "three": 3, "trio": 3, "third player": 3,
        "4": 4, "four": 4,
    },
    "squashLevel": {
        "beginner": "beginner", "novice": "beginner",
        "intermediate": "intermediate", "medium": "intermediate",
        "advanced": "advanced", "expert": "advanced", "pro": "advanced", "professional": "advanced",
    },
    "intensity": {
        "low": "low",
        "medium": "medium", "moderate": "medium",
        "high": "high", "intense": "high", "hard": "high", "extremely high": "high"
    },
    "fitness": {
        "low": "low",
        "medium": "medium", "moderate": "medium",
        "high": "high", "intense": "high", "hard": "high", "extremely high": "high"
    },
    "duration": {
        "10minutes": 10, "10 minutes": 10, "10min": 10, "10 min": 10,
        "20minutes": 20, "20 minutes": 20, "20min": 20, "20 min": 20,
        "30minutes": 30, "30 minutes": 30, "30min": 30, "30 min": 30,
        "45minutes": 45, "45 minutes": 45, "45min": 45, "45 min": 45,
        "60minutes": 60, "60 minutes": 60, "60min": 60, "60 min": 60, "1 hour": 60, "an hour": 60,
        "90minutes": 90, "90 minutes": 90, "90min": 90, "90 min": 90, "hour and a half": 90,
    },
    "shots": {
        "drive": "drive",
        "cross": "cross",
        "drop": "drop", "drops": "drop", "counter drop": "drop", "cross drop": "drop", "straight drop": "drop",
        "boast": "boast", "boasts": "boast", "trickle boast": "boast", "reverse boast": "boast", "2-wall boast": "boast", "3-wall boast": "boast",
        "lob": "lob", "lobs": "lob", "straight lob": "lob", "cross lob": "lob",
        "volley": "volley", "volleys": "volley", "no bounce": "volley",
        "kill": "kill", "straight kill": "kill", "cross kill": "kill",
        "serve": "serve", "serves": "serve",
        "nick": "nick", "nicks": "nick",
        "flick": "flick", "flicks": "flick",
    },
    "shotSide": {
        "forehand": "forehand", "fh": "forehand",
        "backhand": "backhand", "bh": "backhand",
        "both": "both", "either": "both", "forehand and backhand": "both", "backhand and forehand": "both", "fh and bh": "both", "bh and fh": "both",
    },
    "movement": {
        "front": "front",
        "middle": "middle",
        "back": "back",
        "sideways": "sideways", "lateral": "sideways", "side-to-side": "sideways", "side to side": "sideways",
        "diagonal": "diagonal",
        "multi-directional": "multi-directional", "all over the court": "multi-directional",
    },
}

# --- General-to-Specific Mappings ---
# Used by hierarchical boosting logic to broaden general terms to specific variations.
GENERAL_SHOT_TYPES = {
    "drive": ["drive", "deep drive", "hard drive", "straight drive", "volley drive", "volley deep drive", "volley hard drive", "volley straight drive"],
    "cross": ["cross", "cross-court", "cross court", "cross lob", "lob cross", "deep cross", "cross deep", "cross wide", "wide cross", "cross down the middle", "hard cross", "volley cross", "volley hard cross", "volley cross lob", "volley cross-court nick", "cross-court nick"],
    "drop": ["drop", "counter drop", "cross drop", "straight drop", "volley cross drop", "volley straight drop"],
    "boast": ["boast", "2-wall boast", "3-wall boast", "volley 2-wall boast", "volley 3-wall boast", "trickle boast", "reverse boast", "volley reverse boast"],
    "lob": ["lob", "straight lob", "cross lob", "volley straight lob", "volley cross lob"],
    "volley": ["volley", "volley drop", "volley drive", "volley cross", "volley lob", "volley flick", "volley 2-wall boast", "volley 3-wall boast", "volley reverse boast", "volley deep drive", "volley hard drive", "volley straight drive", "volley cross drop", "volley straight drop", "volley cross kill", "volley straight kill", "volley straight lob", "volley cross-court nick", "volley hard cross"],
    "kill": ["straight kill", "volley straight kill", "cross kill", "volley cross kill"],

    # missing info
    "serve": ["lob serve"],
    "nick": ["cross-court nick", "volley cross-court nick"],
    "flick": ["flick", "volley flick"],
}

# Dictionary to easily retrieve specific maps by name (for dynamic access in score_document)
SPECIFIC_MAPS = {
    "GENERAL_SHOT_TYPES": GENERAL_SHOT_TYPES,
    # Add other specific maps here if you create them for other fields
}

# --- Field Scoring Configuration ---
# Defines the scoring method, base weights, and specific parameters for each field.
FIELD_SCORING_CONFIG = {
    "type": {"method": "_score_exact_match_field", "base_weight": 3.0},
    "participants": {"method": "_score_exact_match_field", "base_weight": 3.0},
    "squashLevel": {"method": "_score_exact_match_field", "base_weight": 2.0},
    "intensity": {
        "method": "_score_inferred_categorical_match_field",
        "base_weight": 1.0,
        "secondary_match_multiplier": 0.5,
        "field_name_for_secondary_standardisation": "intensity"
    },
    "duration": {"method": "_score_numerical_range_field", "base_weight": 1.5, "tolerance": 20},
    "shots": {
        "method": "_score_hierarchical_boost_field",
        "base_weight": 2.0,
        "primary_boost_weight": 6.0,
        "secondary_boost_weight": 5.0,
        "field_name_for_standardisation": "shots",
        "general_to_specific_map_name": "GENERAL_SHOT_TYPES"
    },
    "shotSide": {"method": "_score_list_overlap_field", "base_weight": 1.0},
    "movement": {"method": "_score_list_overlap_field", "base_weight": 1.0},
    "fitness": {"method": "_score_exact_match_field", "base_weight": 0.5},
}