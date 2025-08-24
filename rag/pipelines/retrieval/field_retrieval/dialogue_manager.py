# field_retrieval/dialogue_manager.py
from .user_desires import (
    parse_squash_level,
    parse_participants,
    parse_intensity,
    parse_duration,
    parse_shot_side,
    parse_movement,
    SYNONYM_MAP
)


# We define which fields are worth asking about and how to ask.
# We can also provide examples from the synonym map to help the user.
ASKABLE_FIELDS = {
    'squashLevel': {
        'question': "What is your desired squash level?",
        'examples': ['beginner', 'intermediate', 'advanced']
    },
    'participants': {
        'question': "How many participants?",
        'examples': ['1', '2']
    },
    'intensity': {
        'question': "What intensity level are you looking for?",
        'examples': ['low', 'medium', 'high']
    },
    'duration': {
        'question': "Roughly how long should the session be (in minutes)?",
        'examples': ['45', '60', '90']
    },
    'shotSide': {
        'question': "Any preference for shot side?",
        'examples': ['forehand', 'backhand', 'both']
    },
}

# Map fields to their dedicated parsing functions from user_desires.py
PARSER_MAP = {
    'squashLevel': parse_squash_level,
    'participants': parse_participants,
    'intensity': parse_intensity,
    'duration': parse_duration,
    'shotSide': parse_shot_side,
    'movement': parse_movement,
}


def refine_user_desires(initial_desires: dict) -> dict:
    """
    Engages the user in a dialogue to fill in missing metadata fields.

    Args:
        initial_desires: The dictionary of desires parsed from the initial prompt.

    Returns:
        A new dictionary of desires enriched with the user's explicit
        answers or 'no_preference' for skipped fields.
    """
    final_desires = initial_desires.copy()

    print("\n--- Let's refine your search (press Enter to skip) ---")
    for field, config in ASKABLE_FIELDS.items():
        if field not in final_desires:
            # Construct the question with examples
            question = config['question']
            if config.get('examples'):
                examples_str = ", ".join(config['examples'])
                question_full = f"{question} (e.g., {examples_str})"
            else:
                question_full = question

            answer = input(f"Q: {question_full}\nA: ")

            if not answer.strip():
                # User skipped, so we record 'no_preference'
                final_desires[field] = 'no_preference'
            else:
                # User gave an answer, so we parse it using the correct function
                parser_func = PARSER_MAP.get(field)
                if parser_func:
                    # Only pass 'allowed_durations' to the 'parse_duration' function
                    if field == 'duration':
                        # We pass None here to avoid snapping to buckets yet
                        parsed_value = parser_func(answer, allowed_durations=None)
                    else:
                        # All other parsers only take one argument
                        parsed_value = parser_func(answer)

                    if parsed_value:
                        final_desires[field] = parsed_value
                    else:
                        # If parsing fails, treat as no preference
                        print(f"(Could not understand '{answer}', skipping this field.)")
                        final_desires[field] = 'no_preference'

    print("-" * 20)
    return final_desires