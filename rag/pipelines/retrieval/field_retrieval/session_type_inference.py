import random
from .user_desires import parse_type


def infer_session_type(query: str) -> str:
    """
    Infers the session type from the query using the robust user_desires parser.
    If no type can be inferred, it falls back to a random choice.

    Args:
        query: The user's text prompt.

    Returns:
        The canonical session type (e.g., 'drill', 'mix').
    """
    # Try to infer using the powerful parser
    inferred_type = parse_type(query)
    if inferred_type:
        print(f"   🤖 Inferred session type: '{inferred_type}'")
        return inferred_type

    # If no type is found, fall back to a random choice
    else:
        random_type = random.choice(["conditioned_game", "drill", "mix"])
        print(f"   🤔 Could not infer type, choosing random: '{random_type}'")
        return random_type