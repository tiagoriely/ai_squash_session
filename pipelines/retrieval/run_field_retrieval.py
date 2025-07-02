# evaluation/run_field_retrieval.py

import re
import json
from pathlib import Path

# Assuming these PATTERNS dictionaries are defined somewhere accessible,
# e.g., imported from corpus_tools.py or redefined here if standalone.
# For simplicity, I'm including them here. In a real project, you'd likely
# import them to avoid duplication.

# Define patterns for all fields (duplicated for self-contained file, consider importing)
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

SHOT_PATTERNS = {
    "drive": re.compile(r"\bdrive\b", re.I),
    "cross": re.compile(r"\bcross\b", re.I),
    "lob": re.compile(r"\blob\b", re.I),
    "drop": re.compile(r"\bdrop\b", re.I),
    "boast": re.compile(r"\bboast\b", re.I),
}

SHOT_SIDE_PATTERNS = {
    "forehand": re.compile(r"\bforehand\b|\bFH\b", re.I),
    "backhand": re.compile(r"\bbackhand\b|\bBH\b", re.I),
}

DURATION_PATTERNS = {
    "10": re.compile(r"\b10\s*min(?:ute)?s?\b", re.I),
    "15": re.compile(r"\b15\s*min(?:ute)?s?\b", re.I),
    "30": re.compile(r"\b30\s*min(?:ute)?s?\b", re.I),
    "45": re.compile(r"\b45\s*min(?:ute)?s?\b", re.I),
    "60": re.compile(r"\b60\s*min(?:ute)?s?\b", re.I),
    "90": re.compile(r"\b90\s*min(?:ute)?s?\b", re.I),
}


# Re-use the extract_field_values from corpus_tools.py
def extract_field_values(text: str, patterns: dict, allow_multiple: bool = False, return_first_match: bool = False):
    """
    Extracts values for a given field based on provided patterns.
    (Copied from corpus_tools.py to make this file self-contained for demonstration)
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

    return user_preferences


def score_document(document: dict, user_desires: dict) -> float:
    """
    Scores a single document from the knowledge base against user desires.
    """
    score = 0.0

    # Define weights for each field (you can tune these)
    weights = {
        'type': 1.0,
        'participants': 1.5,
        'squashLevel': 2.0,
        'intensity': 1.5,
        'duration': 1.0,
        'shots': 1.2,  # Each matched shot adds to the score
        'shotSide': 0.8,  # Each matched side adds to the score
        'fitness': 0.5  # Less critical if not explicitly requested
    }

    # Match 'type'
    if user_desires.get('type') and document.get('type') == user_desires['type']:
        score += weights['type']
    elif user_desires.get('type') == 'drill' and document.get('type') == 'mix':
        score += weights['type'] * 0.5  # Partial credit for 'mix' if 'drill' is desired

    # Match 'participants'
    if user_desires.get('participants') and document.get('participants') == user_desires['participants']:
        score += weights['participants']

    # Match 'squashLevel'
    if user_desires.get('squashLevel') and document.get('squashLevel') == user_desires['squashLevel']:
        score += weights['squashLevel']

    # Match 'intensity'
    # Prioritize direct intensity match, then fallback to fitness if intensity isn't in doc or user prompt
    if user_desires.get('intensity') and document.get('intensity') == user_desires['intensity']:
        score += weights['intensity']
    elif user_desires.get('intensity') and document.get('fitness'):
        # Simple mapping for fitness to intensity if a direct intensity match isn't found
        # This part assumes a loose equivalence, refine as needed for your data.
        fitness_to_intensity = {
            'low': 'low', 'medium': 'medium', 'intermediate': 'medium',
            'high': 'high', 'extremely_high': 'high'
        }
        if fitness_to_intensity.get(document['fitness'].lower()) == user_desires['intensity'].lower():
            score += weights['intensity'] * 0.7  # Partial credit

    # Match 'duration' (fuzzy match using a small tolerance, e.g., +/- 10 min)
    if user_desires.get('duration') and document.get('duration'):
        try:
            # Extract only digits from duration strings for comparison
            user_duration_val = int(re.search(r'\d+', user_desires['duration']).group())
            doc_duration_val = int(re.search(r'\d+', document['duration']).group())

            # Allow for a difference of up to 10 minutes (adjust as needed)
            if abs(user_duration_val - doc_duration_val) <= 10:
                # Award score, giving higher points for closer matches
                score += weights['duration'] * (1 - (abs(user_duration_val - doc_duration_val) / user_duration_val))
        except (ValueError, AttributeError):
            pass  # Handle cases where duration isn't perfectly parsable

    # Match 'shots' (check for overlap)
    if user_desires.get('shots') and document.get('shots'):
        matched_shots = set(user_desires['shots']).intersection(set(document['shots']))
        score += weights['shots'] * len(matched_shots)

    # Match 'shotSide' (check for overlap)
    if user_desires.get('shotSide') and document.get('shotSide'):
        matched_sides = set(user_desires['shotSide']).intersection(set(document['shotSide']))
        score += weights['shotSide'] * len(matched_sides)

    # Consider 'fitness' if user specifically asks for it and it's not covered by 'intensity' mapping
    if user_desires.get('fitness') and document.get('fitness') == user_desires['fitness']:
        score += weights['fitness']

    return score


# Example Usage and Retrieval Logic (for demonstration purposes)
if __name__ == "__main__":
    KB_PATH = Path("data/my_kb.jsonl")  # Adjust path if running from different location

    if not KB_PATH.exists():
        print(f"Error: Knowledge base file not found at {KB_PATH}. Please run corpus_tools.py first.")
        exit()

    # Load your knowledge base
    knowledge_base = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            knowledge_base.append(json.loads(line))

    # Example user prompt
    user_prompt = "I want an advanced drill for 2 players focusing on crosses and lobs with medium intensity lasting about 45 minutes."
    user_desires = parse_user_prompt(user_prompt)
    print(f"\nUser Desires: {user_desires}")

    # Score all documents against the user's desires
    scored_documents = []
    for doc in knowledge_base:
        s = score_document(doc, user_desires)
        scored_documents.append((s, doc))

    # Sort by score (descending) and get top N
    # Filter out documents with a score of 0 if they don't match any criteria
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    top_n_documents = [doc for score, doc in scored_documents if score > 0][:5]  # Get top 5 relevant documents

    print("\n--- Top Relevant Documents ---")
    if top_n_documents:
        for i, doc in enumerate(top_n_documents):
            score = next(s for s, d in scored_documents if d['id'] == doc['id'])  # Get score back
            print(f"Rank {i + 1} (Score: {score:.2f}):")
            print(f"  ID: {doc.get('id')}, Source: {doc.get('source')}")
            print(
                f"  Type: {doc.get('type')}, Participants: {doc.get('participants')}, Level: {doc.get('squashLevel')}")
            print(f"  Intensity: {doc.get('intensity')}, Duration: {doc.get('duration')}")
            print(f"  Shots: {doc.get('shots')}, Shot Side: {doc.get('shotSide')}")
            # print(f"  Contents (first 200 chars): {doc.get('contents', '')[:200]}...") # Optional: print snippet
            print("-" * 20)
    else:
        print("No relevant documents found for the given prompt.")


    if top_n_documents:
        # For simplicity, let's assume the "generated session" is just the top document's contents
        # In a real RAG system, this would be the actual text generated by your LLM based on retrieved docs.
        generated_session_content = top_n_documents[0]['contents']
        generated_session_fields = top_n_documents[0]  # Using the top doc's fields as "generated fields"

        print(f"User Desires: {user_desires}")

        # Field Fulfillment Rate
        fulfilled_fields = 0
        total_requested_fields = 0

        # Define the fields you want to check for fulfillment
        fields_to_check = ['type', 'participants', 'squashLevel', 'intensity', 'duration', 'shots', 'shotSide']

        for field in fields_to_check:
            user_val = user_desires.get(field)
            generated_val = generated_session_fields.get(field)

            if user_val:  # If user requested this field
                total_requested_fields += 1
                if isinstance(user_val, list):  # For multi-value fields like shots, shotSide
                    if set(user_val).issubset(set(generated_val or [])):  # Check if all requested are present
                        fulfilled_fields += 1
                elif field == 'duration':  # Special handling for duration range
                    try:
                        user_dur = int(re.search(r'\d+', user_val).group())
                        gen_dur = int(re.search(r'\d+', generated_val).group())
                        if abs(user_dur - gen_dur) <= 10:
                            fulfilled_fields += 1
                    except (ValueError, AttributeError):
                        pass
                elif generated_val == user_val:  # For exact match fields
                    fulfilled_fields += 1

        fulfillment_rate = (fulfilled_fields / total_requested_fields) * 100 if total_requested_fields > 0 else 0
        print(
            f"\nAutomated Field Fulfillment Rate: {fulfilled_fields}/{total_requested_fields} ({fulfillment_rate:.2f}%)")

        # You can add more automated metrics here:
        # - Keyword presence/density in `generated_session_content` for specific shots/sides.
        # - Programmatic check for warm-up/session structure within contents (more complex).
    else:
        print("Cannot perform automated generation evaluation: No top documents found.")