# retrieval/field_matcher.py

import re
import json
from pathlib import Path

# Define patterns for all fields
SESSION_TYPE_PATTERNS = {
    "drill": re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)\b|\bgame\b", re.I),
    "solo": re.compile(r"\bsolo\b", re.I),
    "ghosting": re.compile(r"\bghosting\b", re.I),
}

PARTICIPANTS_PATTERNS = {
    "1": re.compile(r"\b1(?:\s*player)?s?\b|\bsolo\b|\bone\s*player\b", re.I),
    "2": re.compile(r"\b2(?:\s*player)?s?\b|\btwo\s*players\b", re.I),
    "3": re.compile(r"\b3(?:\s*player)?s?\b|\bthree\s*players\b", re.I),
    "4": re.compile(r"\b4(?:\s*player)?s?\b|\bfour\s*players\b", re.I),
}

SQUASH_LEVEL_PATTERNS = {
    "beginner": re.compile(r"\bbeginner\b", re.I),
    "intermediate": re.compile(r"\bintermediate\b", re.I),
    "advanced": re.compile(r"\badvanced\b", re.I),
    "professional": re.compile(r"\bprofessional\b", re.I),
}

FITNESS_PATTERNS = {
    "low": re.compile(r"\blow\s*fitness\b|\blow\s*intensity\b", re.I),
    "medium": re.compile(r"\bmedium\s*fitness\b|\bmedium\s*intensity\b", re.I),
    "high": re.compile(r"\bhigh\s*fitness\b|\bhigh\s*intensity\b", re.I),
    "extremely_high": re.compile(r"\bextremely\s*high\s*fitness\b|\bextremely\s*high\s*intensity\b", re.I),
    "intermediate": re.compile(r"\bintermediate\s*fitness\b|\bintermediate\s*intensity\b", re.I),
}

INTENSITY_PATTERNS = {
    "low": re.compile(r"\blow\s*intensity\b", re.I),
    "medium": re.compile(r"\bmedium\s*intensity\b", re.I),
    "high": re.compile(r"\bhigh\s*intensity\b", re.I),
}

SPECIFICSHOT_PATTERNS = {
    # ─── Boasts ─────────────────────────────────────────────────────────────
    "2-wall boast":        re.compile(r"\b2[-\s]*wall\s*boast\b", re.I),
    "3-wall boast":        re.compile(r"\b3[-\s]*wall\s*boast\b", re.I),
    "trickle boast":       re.compile(r"\btrickle\s*boast\b",   re.I),
    "reverse boast":       re.compile(r"\breverse\s*boast\b",   re.I),

    # volley + boast
    "volley 2-wall boast": re.compile(r"\bvolley\s*2[-\s]*wall\s*boast\b", re.I),
    "volley 3-wall boast": re.compile(r"\bvolley\s*3[-\s]*wall\s*boast\b", re.I),
    "volley reverse boast":re.compile(r"\bvolley\s*reverse\s*boast\b",     re.I),

    # ─── Drops ──────────────────────────────────────────────────────────────
    "counter drop":        re.compile(r"\bcounter\s*drop\b",     re.I),
    "cross drop":          re.compile(r"\bcross\s*drop\b",       re.I),
    "straight drop":       re.compile(r"\bstraight\s*drop\b",    re.I),
    "volley cross drop":   re.compile(r"\bvolley\s*cross\s*drop\b",   re.I),
    "volley straight drop":re.compile(r"\bvolley\s*straight\s*drop\b", re.I),

    # ─── Drives / deep shots / kills ───────────────────────────────────────
    "deep drive":          re.compile(r"\bdeep\s*drive\b",       re.I),
    "hard drive":          re.compile(r"\bhard\s*drive\b",       re.I),
    "straight drive":      re.compile(r"\bstraight\s*drive\b",   re.I),
    "volley deep drive":   re.compile(r"\bvolley\s*deep\s*drive\b",   re.I),
    "volley hard drive":   re.compile(r"\bvolley\s*hard\s*drive\b",   re.I),
    "volley straight drive":re.compile(r"\bvolley\s*straight\s*drive\b", re.I),
    "straight kill":       re.compile(r"\bstraight\s*kill\b",    re.I),
    "cross kill":          re.compile(r"\bcross\s*kill\b",       re.I),
    "volley cross kill":   re.compile(r"\bvolley\s*cross\s*kill\b",   re.I),
    "volley straight kill":re.compile(r"\bvolley\s*straight\s*kill\b", re.I),

    # ─── Cross-variants ────────────────────────────────────────────────────
    "cross lob":           re.compile(r"\bcross\s*lob\b",        re.I),
    "lob cross":           re.compile(r"\blob\s*cross\b",        re.I),
    "cross wide":          re.compile(r"\bcross\s*wide\b",       re.I),
    "cross down the middle":re.compile(r"\bcross[-\s]*down\s*the\s*middle\b", re.I),
    "cross-court nick":    re.compile(r"\bcross[-\s]*court\s*nick\b", re.I),
    "hard cross":          re.compile(r"\bhard\s*cross\b",       re.I),
    "volley cross":        re.compile(r"\bvolley\s*cross\b",     re.I),
    "volley hard cross":   re.compile(r"\bvolley\s*hard\s*cross\b",  re.I),
    "volley cross lob":    re.compile(r"\bvolley\s*cross\s*lob\b",   re.I),
    "volley cross-court nick":
                          re.compile(r"\bvolley\s*cross[-\s]*court\s*nick\b", re.I),

    # ─── Lobs / flicks ─────────────────────────────────────────────────────
    "straight lob":        re.compile(r"\bstraight\s*lob\b",     re.I),
    "volley straight lob": re.compile(r"\bvolley\s*straight\s*lob\b", re.I),
    "flick":               re.compile(r"\bflick\b",              re.I),
    "volley flick":        re.compile(r"\bvolley\s*flick\b",     re.I),

    # (keep any originals that weren’t duplicates)
    "volley drop":         re.compile(r"\bvolley\s*drop\b",      re.I),
    "volley drive":        re.compile(r"\bvolley\s*drive\b",     re.I),

    # ─── Ghosting ──────────────────────────────────────────────────────────
    "3-step ghosting": re.compile(r"\b3[-\s]*step(s)?\s*ghost(ing|s)?\b", re.I),
    "2-step ghosting": re.compile(r"\b2[-\s]*step(s)?\s*ghost(ing|s)?\b", re.I),
    "1-step ghosting": re.compile(r"\b1[-\s]*step(s)?\s*ghost(ing|s)?\b", re.I),
}

SHOT_PATTERNS = {
    "drive":   re.compile(r"\bdrives?\b",  re.I),
    "cross":   re.compile(r"\bcross(?:es)?\b", re.I),
    "lob":     re.compile(r"\blobs?\b",   re.I),
    "drop":    re.compile(r"\bdrops?\b",  re.I),
    "boast":   re.compile(r"\bboasts?\b", re.I),
    "volley":  re.compile(r"\bvolleys?\b",re.I),
    "serve":   re.compile(r"\bserves?\b", re.I),
}

SHOT_SIDE_PATTERNS = {
    "forehand": re.compile(r"\bforehand\b|\bFH\b", re.I),
    "backhand": re.compile(r"\bbackhand\b|\bBH\b", re.I),
}

# Add these field to knowledge base and field retrieval
MOVEMENT_PATTERNS = {
    "front":  re.compile(r"\b(front|fore|forward)\s*(?:corners?)?\b", re.I),
    "middle": re.compile(r"\b(mid(?:dle)?|centre|center)\s*(?:corners?)?\b", re.I),
    "back":   re.compile(r"\b(back)\s*(?:corners?)?\b", re.I),
}

DURATION_PATTERNS = {
    "10": re.compile(r"\b10\s*min(?:ute)?s?\b", re.I),
    "15": re.compile(r"\b15\s*min(?:ute)?s?\b", re.I),
    "30": re.compile(r"\b30\s*min(?:ute)?s?\b", re.I),
    "45": re.compile(r"\b45\s*min(?:ute)?s?\b", re.I),
    "60": re.compile(r"\b60\s*min(?:ute)?s?\b", re.I),
    "90": re.compile(r"\b90\s*min(?:ute)?s?\b", re.I),
}


def extract_field_values(text: str, patterns: dict, allow_multiple: bool = False, return_first_match: bool = False):
    """
    Extracts values for a given field based on provided patterns.
    """
    matched_values = []
    for value, pattern in patterns.items():
        if pattern.search(text):
            matched_values.append(value)

    if allow_multiple:
        return matched_values if matched_values else []
    else:
        if len(matched_values) == 1:
            return matched_values[0]
        if len(matched_values) > 1:
            if return_first_match:
                return matched_values[0]
            return "mix"
        return "unknown"


def parse_user_prompt(prompt: str):
    """
    Parses a user prompt to extract desired field values.
    """
    user_preferences = {}

    user_preferences['type'] = extract_field_values(prompt, SESSION_TYPE_PATTERNS)
    user_preferences['participants'] = extract_field_values(prompt, PARTICIPANTS_PATTERNS, return_first_match=True)
    user_preferences['squashLevel'] = extract_field_values(prompt, SQUASH_LEVEL_PATTERNS)
    user_preferences['fitness'] = extract_field_values(prompt, FITNESS_PATTERNS)
    user_preferences['intensity'] = extract_field_values(prompt, INTENSITY_PATTERNS)
    user_preferences['duration'] = extract_field_values(prompt, DURATION_PATTERNS, return_first_match=True)
    user_preferences['shots'] = extract_field_values(prompt, SHOT_PATTERNS, allow_multiple=True)
    user_preferences['shotSide'] = extract_field_values(prompt, SHOT_SIDE_PATTERNS, allow_multiple=True)
    # ADDED specificShots parsing
    user_preferences['specificShots'] = extract_field_values(prompt, SPECIFICSHOT_PATTERNS, allow_multiple=True)
    user_preferences['movement'] = extract_field_values(prompt, MOVEMENT_PATTERNS, allow_multiple=True)

    return user_preferences


def score_document(document: dict, user_desires: dict) -> float:
    """
    Scores a single document from the knowledge base against user desires based on defined fields.
    """
    score = 0.0

    # Define weights for each field (you can tune these)
    weights = {
        'type': 1.0,
        'participants': 1.5,
        'squashLevel': 2.0,
        'intensity': 1.5,
        'duration': 1.0,
        'shots': 1.2,
        'shotSide': 0.8,
        'fitness': 0.5,
        'specificShots': 1.8, # Added weight for specificShots, give it a higher weight
        'movement': 1.0,
    }

    # Match 'type'
    if user_desires.get('type') and document.get('type') == user_desires['type']:
        score += weights['type']
    elif user_desires.get('type') == 'drill' and document.get('type') == 'mix':
        score += weights['type'] * 0.5

    # Match 'participants'
    if user_desires.get('participants') and document.get('participants') == user_desires['participants']:
        score += weights['participants']

    # Match 'squashLevel'
    if user_desires.get('squashLevel') and document.get('squashLevel') == user_desires['squashLevel']:
        score += weights['squashLevel']

    # Match 'intensity'
    if user_desires.get('intensity') and document.get('intensity') == user_desires['intensity']:
        score += weights['intensity']
    elif user_desires.get('intensity') and document.get('fitness'):
        fitness_to_intensity = {
            'low': 'low', 'medium': 'medium', 'intermediate': 'medium',
            'high': 'high', 'extremely_high': 'high'
        }
        if fitness_to_intensity.get(document['fitness'].lower()) == user_desires['intensity'].lower():
            score += weights['intensity'] * 0.7

    # Match 'duration' (fuzzy match using a small tolerance)
    if user_desires.get('duration') and document.get('duration'):
        try:
            user_duration_val = int(re.search(r'\d+', user_desires['duration']).group())
            doc_duration_val = int(re.search(r'\d+', document['duration']).group())

            if abs(user_duration_val - doc_duration_val) <= 10:
                score += weights['duration'] * (1 - (abs(user_duration_val - doc_duration_val) / user_duration_val))
        except (ValueError, AttributeError):
            pass

    # Match 'shots' (check for overlap)
    if user_desires.get('shots') and document.get('shots'):
        matched_shots = set(user_desires['shots']).intersection(set(document['shots']))
        score += weights['shots'] * len(matched_shots)

    # Match 'shotSide' (check for overlap)
    if user_desires.get('shotSide') and document.get('shotSide'):
        matched_sides = set(user_desires['shotSide']).intersection(set(document['shotSide']))
        score += weights['shotSide'] * len(matched_sides)

    # Match 'specificShots' (check for overlap)
    if user_desires.get('specificShots') and document.get('specificShots'):
        matched_specific_shots = set(user_desires['specificShots']).intersection(set(document['specificShots']))
        score += weights['specificShots'] * len(matched_specific_shots)

    # Match 'movement' (check for overlap)
    if user_desires.get('movement') and document.get('movement'):
        matched_shots = set(user_desires['movement']).intersection(set(document['movement']))
        score += weights['movement'] * len(matched_shots)


    # Consider 'fitness' if user specifically asks for it
    if user_desires.get('fitness') and document.get('fitness') == user_desires['fitness']:
        score += weights['fitness']

    return score