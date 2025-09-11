# tests/evaluation_tests/test_check_complex_structure.py

import pytest
from evaluation.utils.context_metrics.structure_metrics import check_complex_structure

# --- Test Cases for Valid Structures ---

def test_valid_plan_single_activity():
    """Checks a valid plan with one Warm-up and one correct Activity block."""
    plan = """
    ### Warm-up ###
    - Some warm-up exercise.
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    """
    assert check_complex_structure(plan) is True

def test_valid_plan_multiple_activities():
    """Checks a valid plan with multiple correct Activity blocks."""
    plan = """
    ### Warm-up ###
    - Some warm-up exercise.
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    ### Activity Block 2 ###
    * Another exercise.
    * And a second one.
    - Rest: 1.5 min
    """

    plan2 = """
        ### Warm-up ###
        - Some warm-up exercise.
        ### Activity Block 1 ###
        1. First exercise.
        2. Second exercise.
        3. Rest: 1.5 minutes
        ### Activity Block 2 ###
        Another exercise.
        And a second one.
        Rest: 1.5 min
        """

    plan3 = """
            ### Warm-up ###
            - Some warm-up exercise.
            ### Activity Block 1 ###
            – First exercise.
            – Second exercise.
            – Rest: 1.5 minutes
            ### Activity Block 2 ###
            • Another exercise.
            • And a second one.
            - Rest: 1.5 min
            """
    assert check_complex_structure(plan) is True
    assert check_complex_structure(plan2) is True
    assert check_complex_structure(plan3) is True

def test_valid_plan_with_surrounding_noise():
    """
    Ensures the check ignores extra blocks before and after the core structure.
    """
    plan = """
    ### Pre-Session Notes ###
    - Remember to hydrate.
    ### Warm-up ###
    - Some warm-up exercise.
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    ### Cool-down ###
    - Stretching.
    """
    assert check_complex_structure(plan) is True

# --- Test Cases for Invalid Macro-Structure (Block Order/Presence) ---

def test_invalid_no_warmup():
    """Fails if the mandatory Warm-up block is missing."""
    plan = """
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    """
    assert check_complex_structure(plan) is False

def test_invalid_no_activity_after_warmup():
    """Fails if no Activity block follows the Warm-up."""
    plan = """
    ### Warm-up ###
    - Some warm-up exercise.
    ### Cool-down ###
    - Stretching.
    """
    assert check_complex_structure(plan) is False

def test_invalid_wrong_order():
    """Fails if an Activity block appears before the first Warm-up."""
    plan = """
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    ### Warm-up ###
    - Some warm-up exercise.
    """
    assert check_complex_structure(plan) is False

# --- Test Cases for Invalid Micro-Structure (Internal Pattern) ---

def test_invalid_micro_wrong_exercise_count():
    """Fails if an Activity block has the wrong number of exercises."""
    plan = """
    ### Warm-up ###
    - Some warm-up exercise.
    ### Activity Block 1 ###
    1. Just one exercise.
    - Rest: 1.5 minutes
    """
    assert check_complex_structure(plan) is False


# def test_invalid_micro_wrong_rest_order():
#     """Fails if the rest period appears before the exercises."""
#     plan = """
#     ### Warm-up ###
#     - Some warm-up exercise.
#     ### Activity Block 1 ###
#     - Rest: 1.5 minutes
#     1. First exercise.
#     2. Second exercise.
#     """
#     assert check_complex_structure(plan) is False

def test_invalid_one_of_many_activities_is_bad():
    """
    Fails if the plan has multiple Activity blocks but one is invalid.
    """
    plan = """
    ### Warm-up ###
    - Some warm-up exercise.
    ### Activity Block 1 ###
    1. First exercise.
    2. Second exercise.
    - Rest: 1.5 minutes
    ### Activity Block 2 (Invalid) ###
    * Just one exercise in this block.
    - Rest: 1.5 min
    """
    assert check_complex_structure(plan) is False