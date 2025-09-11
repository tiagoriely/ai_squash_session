# tests/evaluation_tests/test_psa_evaluator.py

import pytest
from pathlib import Path
from evaluation.utils.context_metrics.structure_metrics import calculate_psa_flexible

# Pytest fixture to create a temporary EBNF grammar file for tests
@pytest.fixture
def ebnf_grammar(tmp_path: Path) -> Path:
    """
    Creates a temporary, but complete, EBNF grammar for testing purposes.
    """
    # CORRECTED: Added the required Terminal definitions for Lark to parse correctly.
    grammar_content = """
        session: drill_session | mix_session

        drill_session: WARMUP_BLOCK ACTIVITY_BLOCK+
        mix_session: WARMUP_BLOCK ACTIVITY_BLOCK

        // Terminal definitions are mandatory for the Lark parser
        WARMUP_BLOCK: "WARMUP_BLOCK"
        ACTIVITY_BLOCK: "ACTIVITY_BLOCK"

        %import common.WS
        %ignore WS
    """
    p = tmp_path / "test_grammar.ebnf"
    p.write_text(grammar_content)
    return p

# --- Test Cases (No changes needed below) ---

def test_psa_valid_structure(ebnf_grammar):
    """Checks a perfectly valid structure."""
    plan = """
    ### Warm-up ###
    Some text.
    ### Activity Block 1 ###
    Some text.
    ### Activity Block 2 ###
    More text.
    """
    assert calculate_psa_flexible(plan, ebnf_grammar) == 1.0

def test_psa_invalid_order(ebnf_grammar):
    """Checks a plan with the wrong block order."""
    plan = """
    ### Activity Block 1 ###
    Some text.
    ### Warm-up ###
    More text.
    """
    assert calculate_psa_flexible(plan, ebnf_grammar) == 0.0

def test_psa_missing_required_block(ebnf_grammar):
    """Checks a plan missing the mandatory Activity block."""
    plan = """
    ### Warm-up ###
    Some text.
    """
    assert calculate_psa_flexible(plan, ebnf_grammar) == 0.0

def test_psa_flexible_headers(ebnf_grammar):
    """Ensures headers with extra text are parsed correctly."""
    plan = """
    ### Warm-up (10 minutes) ###
    Some text.
    ### Activity Block 1 - Drills ###
    More text.
    """
    assert calculate_psa_flexible(plan, ebnf_grammar) == 1.0

def test_psa_with_extra_illegal_block(ebnf_grammar):
    """Checks that plans with unknown blocks are correctly invalidated."""
    plan = """
    ### Warm-up ###
    Some text.
    ### Activity Block 1 ###
    More text.
    ### Cool-down ###
    Illegal block.
    """
    assert calculate_psa_flexible(plan, ebnf_grammar) == 0.0

def test_psa_no_headers(ebnf_grammar):
    """Checks that a plan with no valid headers fails."""
    plan = "This is a plan with no structure."
    assert calculate_psa_flexible(plan, ebnf_grammar) == 0.0