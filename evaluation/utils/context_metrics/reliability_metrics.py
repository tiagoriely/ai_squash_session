# evaluation/metrics/reliability_metrics.py

import re
from typing import List, Dict, Any


# Placeholder for the LlmEvaluator, which we will build next.
# For now, this allows the code to be written and tested with a mock object.
class LlmEvaluator:
    def check_constraint_alignment(self, human_rule: str, formal_rule: str) -> float:
        # This will be replaced by an actual LLM call in the next step.
        print("--- Mock LLM Call for PCA ---")
        print(f"Human: {human_rule}")
        print(f"Formal: {formal_rule}")
        # For testing, we can return a default value or simple logic.
        return 1.0 if human_rule and formal_rule else 0.0


# It's good practice to handle potential import errors if run standalone
try:
    from rag.parsers.user_query_parser import parse_user_prompt
except ImportError:
    print("Warning: rag.parsers.user_query_parser not found.")
    parse_user_prompt = None


def _parse_plan_metadata(plan_text: str) -> Dict[str, Any]:
    """Helper to extract metadata from the generated plan's text header."""
    metadata = {}
    duration_match = re.search(r"Duration:\s*(\d+)\s*min", plan_text)
    if duration_match:
        metadata["duration"] = int(duration_match.group(1))

    focus_match = re.search(r"Session Focus:\s*(.*)", plan_text)
    if focus_match:
        # A simple placeholder, can be expanded to parse level, side etc. from focus
        if "beginner" in focus_match.group(1).lower():
            metadata["squashLevel"] = "beginner"
        elif "intermediate" in focus_match.group(1).lower():
            metadata["squashLevel"] = "intermediate"
        elif "advanced" in focus_match.group(1).lower():
            metadata["squashLevel"] = "advanced"

    return metadata


def calculate_mas(generated_plan_text: str, user_query: str) -> float:
    """
    Calculates Metadata Alignment Score (MAS).

    Compares the metadata requested in the user query with the metadata
    stated in the generated plan.

    Args:
        generated_plan_text (str): The full text of the generated session plan.
        user_query (str): The original user query.

    Returns:
        float: The ratio of matched metadata fields to requested fields.
    """
    if parse_user_prompt is None:
        raise ImportError("rag.parsers.user_query_parser.parse_user_prompt is required.")

    user_desires = parse_user_prompt(user_query)
    plan_metadata = _parse_plan_metadata(generated_plan_text)

    if not user_desires:
        return 1.0  # Vacuously true if user requested nothing specific

    matched_fields = 0
    for key, value in user_desires.items():
        if key in plan_metadata and str(plan_metadata[key]) == str(value):
            matched_fields += 1

    return matched_fields / len(user_desires)


def calculate_pca(
        generated_plan_text: str,
        retrieved_docs: List[Dict],
        evaluator: LlmEvaluator,  # The LLM judge instance
        exercise_definitions: Dict[str, Any]  # Pre-loaded grammar YAMLs
) -> float:
    """
    Calculates Programmatic Constraint Adherence (PCA).

    Checks if the human-readable rules in the plan accurately reflect the
    formal constraint rules from the source YAMLs of the retrieved exercises.

    Args:
        generated_plan_text (str): The full text of the generated session plan.
        retrieved_docs (List[Dict]): The context docs provided to the generator.
        evaluator (LlmEvaluator): The LLM judge instance for checking alignment.
        exercise_definitions (Dict): A pre-loaded dictionary mapping variant_id to its full YAML definition.

    Returns:
        float: The ratio of correctly represented constraints.
    """
    # 1. Parse exercises and their rules from the generated text
    # This regex captures the exercise name and its associated rule text
    found_exercises = re.findall(r":\s*(?:Drill|Conditioned Game):\s*(.*?)\n\s*\(Rule:\s*(.*?)\)", generated_plan_text,
                                 re.DOTALL)

    if not found_exercises:
        return 1.0  # No rules to check, so no violations.

    total_rules_checked = 0
    correctly_represented_rules = 0

    # Map retrieved doc contents/IDs to their full definitions for easy lookup
    doc_id_map = {doc.get('id'): doc for doc in retrieved_docs}

    for name, human_rule in found_exercises:
        name = name.strip()
        # 2. Find the formal rule from the source definition
        # This part requires a strategy to link the generated name back to a variant_id
        # A simple strategy is to find a doc whose `contents` contains the name.
        # A better strategy would involve having IDs in the generated text or a fuzzy match.
        # For now, let's assume we can find the source variant_id.

        # This is a placeholder for the complex logic of matching name to variant_id
        # In a real scenario, you'd have a robust lookup function here.
        found_variant_id = None
        for variant_id, definition in exercise_definitions.items():
            if name == definition['name']:
                found_variant_id = variant_id
                break

        if found_variant_id:
            formal_rule = exercise_definitions[found_variant_id].get('rules', {}).get('constraint_formal')
            if formal_rule:
                total_rules_checked += 1
                # 3. Use the LLM evaluator to check alignment
                alignment_score = evaluator.check_constraint_alignment(human_rule.strip(), formal_rule)
                if alignment_score > 0.5:  # Using a threshold for YES/NO
                    correctly_represented_rules += 1

    if total_rules_checked == 0:
        return 1.0  # No formal rules were found to check against.

    return correctly_represented_rules / total_rules_checked