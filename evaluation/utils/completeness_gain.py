# evaluators/completeness_gain.py

import re
from pathlib import Path
from typing import List, Dict, Any

# The base class is assumed to be in the same directory or accessible via the python path.
# This matches the pattern from the provided `evaluation/utils/CompletenessGain.py`
from .Metrics import Metrics


class CompletenessGain(Metrics):
    """
    Calculates the Completeness Gain (CG) for structure.

    This metric evaluates how well a generated plan covers key structural
    rules derived from an EBNF grammar, compared to a baseline.
    The core idea is adapted from the RAG Playground paper, focusing on
    structural completeness implied by a grammar rather than semantic
    completeness from source documents[cite: 75, 173].

    A score > 0.5 means the plan covers more structural rules than the baseline.
    A score = 0.5 means equal coverage.
    A score < 0.5 means the plan covers fewer rules.
    """

    def __init__(self, ebnf_grammar_path: str | Path):
        super().__init__()
        self.name = 'Completeness-Gain-Structure'
        self.ebnf_path = Path(ebnf_grammar_path)

        # Mapping from EBNF terminal names to the expected markdown headers in the plan.
        # This makes the checks robust, configurable, and aligned with the plan generation format.
        self.block_mappings = {
            "WARMUP_BLOCK": "Warm-up",
            "ACTIVITY_BLOCK": "Activity",
            "CG_BLOCK": "Conditioned Game",
        }

        if not self.ebnf_path.is_file():
            raise FileNotFoundError(f"EBNF grammar file not found at: {self.ebnf_path}")

        self.key_points = self._extract_key_points_from_ebnf()
        print(
            f"✅ CompletenessGain initialised with {len(self.key_points)} structural points from {self.ebnf_path.name}.")

    def _extract_key_points_from_ebnf(self) -> List[Dict[str, str]]:
        """
        Parses the EBNF file to extract structural rules as key points ('atoms').
        This implementation uses simple, robust regex to identify mandatory blocks
        based on their presence in rule definitions.
        """
        points = []
        grammar_text = self.ebnf_path.read_text()

        # Consolidate all rule bodies to check for terminal presence globally.
        rules = re.findall(r"\w+\s*:\s*(.*)", grammar_text)
        all_rule_bodies = " ".join(rules)

        for terminal, plan_text in self.block_mappings.items():
            # Rule 1: Check if a block type is mandatory in any rule.
            if terminal in all_rule_bodies:
                points.append({
                    "text": f"Session must contain a '{plan_text}' block.",
                    "check_text": f"### {plan_text}",  # Checks for the markdown header
                    "type": "presence"
                })

            # Rule 2: Check for multiplicity (e.g., ACTIVITY_BLOCK+)
            # This is illustrative; the current _check_coverage logic only checks for presence (>=1),
            # but this structure allows for future extension to check for specific counts.
            if f"{terminal}+" in all_rule_bodies:
                points.append({
                    "text": f"Session must contain one or more '{plan_text}' blocks.",
                    "check_text": f"### {plan_text}",
                    "type": "presence_multiple"
                })

        # Normalise the list of atoms by removing duplicates.
        unique_points = []
        seen = set()
        for point in points:
            if point['text'] not in seen:
                unique_points.append(point)
                seen.add(point['text'])

        return unique_points

    def _check_coverage(self, plan_text: str) -> int:
        """
        Checks a given text against the extracted key points.
        This updated version checks if any line *starts with* the required header,
        making it robust to extra details like timings in the header line.
        """
        if not self.key_points:
            return 0

        covered_count = 0
        plan_lines = [line.strip().lower() for line in plan_text.split('\n')]

        for point in self.key_points:
            check_text_lower = point['check_text'].lower()

            # Check if any line in the plan starts with the required header text
            if any(line.startswith(check_text_lower) for line in plan_lines):
                covered_count += 1

        return covered_count

    def get_score(self, **kwargs: Any) -> Dict[str, float | int]:
        """
        Calculates the final Completeness Gain score and its components.

        Args:
            **kwargs: Expects 'generated_plan' (str) and 'baseline_text' (str).

        Returns:
            A dictionary with the CG score and intermediate calculations.
        """
        generated_plan = kwargs.get("generated_plan")
        baseline_text = kwargs.get("baseline_text")

        if not generated_plan or not baseline_text:
            raise ValueError("`get_score` requires 'generated_plan' and 'baseline_text' arguments.")

        k_size = len(self.key_points)
        if k_size == 0:
            return {"completeness_gain": 0.5, "c_resp": 0.0, "c_base": 0.0, "k_size": 0}

        # 1. Calculate coverage for the generated plan (response) and the baseline.
        plan_coverage = self._check_coverage(generated_plan)
        baseline_coverage = self._check_coverage(baseline_text)

        # 2. Normalise scores to a [0, 1] range.
        c_resp = plan_coverage / k_size
        c_base = baseline_coverage / k_size

        # 3. Calculate final Completeness Gain using the paper's normalised formula[cite: 178].
        # A small epsilon is added to prevent division by zero if both coverages are zero.
        denominator = c_resp + c_base + 1e-6
        if c_resp == 0 and c_base == 0:
            cg_score = 0.5  # By definition, if both are empty, their coverage is equal.
        else:
            cg_score = c_resp / denominator

        return {
            "completeness_gain": round(cg_score, 4),
            "c_resp": round(c_resp, 4),    # Normalised coverage of the target plan
            "c_base": round(c_base, 4),    # Normalised coverage of the baseline plan
            "k_size": k_size               # Total number of structural points
        }