import pytest
from rag.parsers.user_query_parser import parse_user_prompt


# Helper function to check subset containment
def _subset(container_list, subset_list):
    c = set(container_list or [])
    s = set(subset_list or [])
    return s.issubset(c)


# Define all your prompts and their expected shot tags
PROMPTS = [
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes volley hard drive and volley straight drive.",
    "Create a training routine that helps with 'Mastering the Boast-Cross-Drive Rally Pattern to Force Maximum Diagonal Court Coverage and Strategic Cross-Court Volley Interception'.",
    #"Create a training routine that helps with 'Strategic Application of Both 2-Wall and 3-Wall Boasts within a Driving Game'.",
    "Design a 60min session to improve my straight drop.",
    "Design a 60min session to improve my straight kill.",
    "Design a 60min session to improve my hard drive.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my deep drive.",
    "Create a training routine that helps with 'Mastering Deep Drive Foundations and Strategic Cross-Court Attack Combinations'.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my hard drive.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my cross deep.",
    "Design a 60min session to improve my deep drive.",
    "I need a Intermediate 45min session for 2 players.",
    "Design a 60min session to improve my cross deep.",
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes 3-step ghosting and 3-step ghosting.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my volley cross.",
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes cross kill and straight drive.",
    "Create a training routine that helps with 'learn to use defensive cross lob to give yourself time to go back on the T'.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my cross drop.",
    "Generate a High intensity session for the ['forehand', 'backhand'] that includes cross drop and volley cross drop.",
    "Create a training routine that helps with 'Mastering Lob-Based Rally Control. Employing Height (lob cross and straight lob) Strategically for Attack and Defense'.",
    "I need a Medium 60min session for 2 players.",
    "Design a 60min session to improve my volley straight drive.",
    "Create a training routine that helps with 'Developing strategic and technical skills required to consistently win rallies while operating under significant court and shot-type restrictions, against an opponent with full court access'.",
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes straight lob and counter drop.",
    "Create a training routine that helps with 'Mastering Foundational Depth, Height, and Alley Control for Beginner/Intermediate Rally Consistency'.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my straight drop.",
    "Create a training routine that helps with 'master your straight drives from the front & front court movement.'.",
    "Design a 60min session to improve my cross lob.",
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes volley cross and volley straight drive.",
    "Design a 60min session to improve my cross drop.",
    "Design a 60min session to improve my cross kill.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my volley hard drive.",
    "I need a Advanced 90min session for 2 players.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my 3-step ghosting.",
    "Design a 60min session to improve my counter drop.",
    "Design a 60min session to improve my volley straight drop.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my deep cross.",
    "Generate a Medium intensity session for the ['backhand'] that includes straight drop and straight lob.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my cross kill.",
    "I need a Beginner 60min session for 2 players.",
    "I need a Advanced 60min session for 2 players.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my straight lob.",
    "Design a 60min session to improve my straight drive.",
    "Generate a Medium intensity session for the ['forehand', 'backhand'] that includes deep cross and straight lob.",
    "Create a training routine that helps with 'Mastering the Strategic Dynamics and Shot Execution in Cross-Court vs. Straight-Court Rallies.'.",
    "My opponent is strong on the attack. Give me a session to practice my defensive game, especially my straight drive.",
    "Create a training routine that helps with 'Mastering Foundational Depth, Height, and Halfcourt Control for Beginner Rally Consistency'.",
    "Design a 90min session to improve my straight lob.",
    "I need a Advanced 45min session for 2 players.",
    "Design a 60min session to improve my deep cross."
]

EXPECTED_SHOTS = [
    {"volley hard drive", "volley straight drive"},
    {"boast", "cross", "drive", "volley"},
    #{"2-wall boast", "3-wall boast", "drive"},
    {"straight drop"},
    {"straight kill"},
    {"hard drive"},
    {"deep drive"},
    {"deep drive", "cross"},
    {"hard drive"},
    {"cross deep"},
    {"deep drive"},
    set(),  # No shot specified
    {"cross deep"},
    set(),  # Ghosting is movement, not shot
    {"volley cross"},
    {"cross kill", "straight drive"},
    {"cross lob"},
    {"cross drop"},
    {"cross drop", "volley cross drop"},
    {"lob cross", "straight lob", "lob"},
    set(),  # No shot specified
    {"volley straight drive"},
    set(),  # Too vague
    {"straight lob", "counter drop"},
    set(),  # Too vague
    {"straight drop"},
    {"straight drive"},
    {"cross lob"},
    {"volley cross", "volley straight drive"},
    {"cross drop"},
    {"cross kill"},
    {"volley hard drive"},
    set(),  # No shot specified
    set(),  # Ghosting is movement, not shot
    {"counter drop"},
    {"volley straight drop"},
    {"deep cross"},
    {"straight drop", "straight lob"},
    {"cross kill"},
    set(),  # No shot specified
    set(),  # No shot specified
    {"straight lob"},
    {"straight drive"},
    {"deep cross", "straight lob"},
    set(),  # Too vague
    {"straight drive"},
    set(),  # Too vague
    {"straight lob"},
    set(),  # No shot specified
    {"deep cross"}
]


@pytest.mark.parametrize(
    "text, expected_shot_set",
    zip(PROMPTS, EXPECTED_SHOTS),
    ids=[f"prompt_{i}" for i in range(len(PROMPTS))]
)
def test_prompt_shot_retrieval(text, expected_shot_set):
    """Test that all prompts retrieve the expected shot tags"""
    result = parse_user_prompt(text, allowed_durations=None)
    retrieved_shots = set(result.get("shots", []))

    # Verify expected shots are present
    missing = expected_shot_set - retrieved_shots
    assert not missing, f"Missing shots: {missing} in prompt: '{text}'"

    # Verify no unexpected wall-specific shots are present
    wall_specifics = {"2-wall boast", "3-wall boast"}
    unexpected = wall_specifics & retrieved_shots
    if not any(w in text for w in ["2-wall", "two-wall", "3-wall", "three-wall"]):
        assert not unexpected, f"Unexpected wall-specific shots: {unexpected} in prompt: '{text}'"