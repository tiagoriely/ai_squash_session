# src/grammar_tools/analysis/sequence_analyser.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple


class SequenceAnalyser:
    """
    Extracts and formats the sequence DSL and its parsed AST from exercises
    within a session plan.
    """

    def __init__(self):
        """Initialises the SequenceAnalyser."""
        pass

    def extract_sequences(self, all_exercises: List[Tuple[Dict[str, Any], Any, str, float]]) -> List[Dict[str, Any]]:
        """
        Iterates through all exercises in a plan and extracts their sequence data.

        The necessary data (sequence_ast and the raw sequence string) is expected
        to be present on the exercise details dictionary, as populated by the loader.

        Args:
            all_exercises: A list of exercise tuples from a session plan.
                           Each tuple contains (details, value, mode, duration).

        Returns:
            A list of dictionaries, where each dictionary contains the identity
            and sequence information for a single exercise.
        """
        extracted_data = []
        for exercise_tuple in all_exercises:
            details = exercise_tuple[0]

            # The loader adds sequence_ast, and the original rules dict is also available
            ast = details.get("sequence_ast")
            raw_dsl = (details.get("rules") or {}).get("sequence")

            # We only add an entry if a sequence was defined for the exercise
            if ast and raw_dsl:
                extracted_data.append({
                    "exercise_family_id": details.get("family_id"),
                    "exercise_variant_id": details.get("variant_id"),
                    "sequence_dsl": raw_dsl,
                    "sequence_ast": ast
                })

        return extracted_data