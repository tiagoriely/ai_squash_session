# src/grammar_tools/dsl_tools/structure_validator.py

from pathlib import Path
from typing import Any, Dict

try:
    from lark import Lark, UnexpectedToken
except ImportError:
    raise ImportError("Lark parser is required for EBNF validation. Please run: pip install lark-parser")


def _convert_session_to_string(session_plan: Dict[str, Any]) -> str:
    """
    Converts a session's block structure into a simple, space-separated string
    that can be parsed by the EBNF grammar.
    """
    block_types = []
    for block in session_plan.get("blocks", []):
        name = block.get("name", "").lower()
        # You can make these mappings as specific as your EBNF requires
        if "warm-up" in name:
            block_types.append("WARMUP_BLOCK")
        elif "conditioned game" in name or "cg" in name:
            block_types.append("CG_BLOCK")
        elif "activity" in name or "drill" in name:
            block_types.append("ACTIVITY_BLOCK")
        # Add other block types if your EBNF defines them

    return " ".join(block_types)


def validate_session_structure(session_plan: Dict[str, Any], ebnf_grammar_path: Path) -> bool:
    """
    Validates a session plan's structure against a given EBNF grammar file.

    Args:
        session_plan: The generated session plan dictionary.
        ebnf_grammar_path: The Path to the .ebnf file for the current grammar profile.

    Returns:
        True if the structure is valid, False otherwise.
    """
    if not ebnf_grammar_path.is_file():
        print(f"⚠️  Warning: EBNF grammar file not found at {ebnf_grammar_path}. Skipping validation.")
        return True  # Fail open if the grammar file is missing

    try:
        grammar_text = ebnf_grammar_path.read_text()
        parser = Lark(grammar_text, start='session')  # 'session' is our top-level rule
    except Exception as e:
        print(f"⚠️  Error loading EBNF grammar: {e}. Skipping validation.")
        return True

    session_string = _convert_session_to_string(session_plan)

    try:
        parser.parse(session_string)
        # print(f"✅ Structure OK: '{session_string}'") # Uncomment for debugging
        return True
    except UnexpectedToken as e:
        print(f"❌ INVALID STRUCTURE: The generated session with blocks '{session_string}'")
        print(f"   does not conform to the EBNF grammar at {ebnf_grammar_path.name}.")
        print(f"   Lark parser error: {e}")
        return False