# evaluation/metrics/structure_metrics.py

import re
import yaml
from pathlib import Path
from typing import Dict, Any

try:
    from lark import Lark, exceptions
    from rag.parsers.user_query_parser import parse_type
except ImportError:
    print("Warning: Required libraries not found. Please ensure lark and rag modules are installed.")
    Lark = None
    parse_type = None


def calculate_psa(
        generated_plan_text: str,
        ebnf_grammar_path: str | Path
) -> float:
    """
    Calculates Programmatic Structure Adherence (PSA) 2.0.

    Parses the high-level block structure of a generated plan against a formal
    EBNF grammar.

    Args:
        generated_plan_text (str): The full text of the generated session plan.
        ebnf_grammar_path (str | Path): Path to the .ebnf file for the expected structure.

    Returns:
        float: 1.0 if the structure is valid, 0.0 otherwise.
    """
    if Lark is None:
        raise ImportError("Lark library is required for PSA calculation.")

    try:
        with open(ebnf_grammar_path, 'r') as f:
            grammar = f.read()

        # Extract the high-level structure by finding block headers
        # This regex finds the headers and normalizes them for the EBNF grammar
        blocks = re.findall(r"### (Warm-up|Activity Block \d+) ###", generated_plan_text)
        structure_string = " ".join([
            "WARMUP_BLOCK" if "Warm-up" in b else "ACTIVITY_BLOCK" for b in blocks
        ])

        if not structure_string:
            return 0.0

        parser = Lark(grammar, start='session')
        parser.parse(structure_string)
        return 1.0
    except (exceptions.LarkError, FileNotFoundError, Exception) as e:
        # If parsing fails or file not found, the structure is not adherent.
        # print(f"PSA Check Failed: {e}") # Optional: for debugging
        return 0.0


def calculate_psa_flexible(
        generated_plan_text: str,
        ebnf_grammar_path: str | Path
) -> float:
    """
    Calculates Programmatic Structure Adherence (PSA) with flexible header parsing.

    This version flexibly identifies headers and then uses the Lark parser to
    validate the sequence against the EBNF grammar. This is robust to minor
    formatting variations and correctly penalises plans with unknown sections.
    """
    if Lark is None:
        raise ImportError("Lark library is required for PSA calculation.")

    try:
        with open(ebnf_grammar_path, 'r') as f:
            grammar = f.read()

        # Flexibly find all markdown headers and normalise them to EBNF terminals
        # This approach is robust to extra text (e.g., timings) and formatting quirks.
        found_blocks = []
        for line in generated_plan_text.split('\n'):
            line_lower = line.strip().lower()
            if line_lower.startswith('### warm-up'):
                found_blocks.append("WARMUP_BLOCK")
            elif line_lower.startswith('### activity'):
                found_blocks.append("ACTIVITY_BLOCK")
            # NOTE: We are intentionally ignoring unknown headers like "cool-down".
            # If an unknown header appears, it won't be added to found_blocks.
            # If the grammar requires more blocks than are found, the parse will fail,
            # correctly marking the structure as invalid.

        structure_string = " ".join(found_blocks)

        if not structure_string:
            return 0.0

        # Use Lark to parse the validated sequence of blocks
        parser = Lark(grammar, start='session')
        parser.parse(structure_string)

        # Final check: does the generated text contain *only* known headers?
        # This prevents a plan with an illegal "Cool-down" from passing.
        all_headers = re.findall(r"### (.*)", generated_plan_text, re.IGNORECASE)
        known_headers = ("warm-up", "activity block")
        for header in all_headers:
            if not any(known in header.lower() for known in known_headers):
                return 0.0  # Found an unknown header, so the structure is invalid.

        return 1.0
    except (exceptions.LarkError, FileNotFoundError):
        return 0.0


def calculate_pda(
        generated_plan_text: str,
        source_structure_template_path: str | Path,
        session_types_config_path: str | Path
) -> float:
    """
    Calculates Programmatic Duration Adherence (PDA).

    Verifies if the total calculated duration of exercises in the plan adheres
    to the overshoot/undershoot rules from its source structure template.

    Args:
        generated_plan_text (str): The full text of the generated session plan.
        source_structure_template_path (str | Path): Path to the source YAML structure file.
        session_types_config_path (str | Path): Path to the sessions_types.yaml config.

    Returns:
        float: 1.0 if the duration is within the allowed range, 0.0 otherwise.
    """
    try:
        with open(source_structure_template_path, 'r') as f:
            structure_template = yaml.safe_load(f)
        with open(session_types_config_path, 'r') as f:
            session_types_config = yaml.safe_load(f)

        # 1. Get rules from the template
        target_duration = structure_template.get('target_duration_minutes', 0)
        overshoot = structure_template.get('total_duration_rules', {}).get('soft_max_overshoot', 0)
        undershoot = structure_template.get('total_duration_rules', {}).get('soft_min_undershoot', 0)
        point_conversion_factor = session_types_config.get('conditioned_game', {}).get('point_duration_minutes', 0.857)

        # 2. Parse the generated text to calculate total activity time
        timed_drills = re.findall(r"•\s*([\d.]+)\s*min:", generated_plan_text)
        point_games = re.findall(r"•\s*([\d.]+)\s*pts:", generated_plan_text)
        rest_periods = re.findall(r"Rest:\s*([\d.]+)\s*min", generated_plan_text)

        total_calculated_duration = 0.0
        total_calculated_duration += sum(float(d) for d in timed_drills)
        total_calculated_duration += sum(float(p) * point_conversion_factor for p in point_games)
        total_calculated_duration += sum(float(r) for r in rest_periods)

        # 3. Check if the calculated duration is within the allowed range
        lower_bound = target_duration - undershoot
        upper_bound = target_duration + overshoot

        if lower_bound <= total_calculated_duration <= upper_bound:
            return 1.0
        else:
            return 0.0

    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        # print(f"PDA Check Failed: {e}") # Optional: for debugging
        return 0.0


def calculate_stc(generated_plan_text: str, user_query: str) -> float:
    """
    Calculates Session Type Consistency (STC).

    Checks if the plan's content (drill, game, mix) is consistent with the
    user's query. If the query is non-specific, any type is considered consistent.

    Args:
        generated_plan_text (str): The full text of the generated session plan.
        user_query (str): The original user query.

    Returns:
        float: 1.0 for consistency, 0.0 otherwise.
    """
    if parse_type is None:
        raise ImportError("rag.parsers.user_query_parser.parse_type is required.")

    expected_type = parse_type(user_query)

    # If user doesn't specify a type, any generated type is consistent.
    if not expected_type or expected_type == "mix":
        return 1.0

    activity_blocks = " ".join(re.findall(r"### Activity Block.*?###(.*?)###", generated_plan_text, re.DOTALL))

    if not activity_blocks:
        activity_blocks = generated_plan_text  # Fallback to whole plan if no blocks found

    has_drills = "min:" in activity_blocks
    has_games = "pts:" in activity_blocks

    actual_type = None
    if has_drills and not has_games:
        actual_type = "drill"
    elif not has_drills and has_games:
        actual_type = "conditioned_game"
    elif has_drills and has_games:
        actual_type = "mix"

    if expected_type == actual_type:
        return 1.0
    else:
        return 0.0