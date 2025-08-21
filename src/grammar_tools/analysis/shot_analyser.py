# src/grammar_tools/shot_analyser.py
from typing import Dict, List, Set


class ShotAnalyser:
    """
    Analyses a session's shots to classify them as primary or secondary
    using a two-tier scoring system based on explicit grammar definitions.
    """

    def __init__(self, field_retrieval_config: Dict):
        # This is no longer needed for scoring but is kept for the extractor's
        # specific-to-general mapping. A future refactor could move this map.
        general_shot_types = field_retrieval_config.get("GENERAL_SHOT_TYPES", {})
        self.specific_to_general_map = {}
        for general, specifics in general_shot_types.items():
            for specific in specifics:
                self.specific_to_general_map[specific] = general

    def classify_shots(self, session_plan: dict, top_n: int = 4) -> Dict[str, List[str]]:
        """
        Scores and classifies shots based on a strict two-tier system.
        """
        shot_scores: Dict[str, float] = {}
        all_specific_shots: Set[str] = set()

        # Iterate through exercises to gather all shots and calculate scores
        for block in session_plan.get("blocks", []):
            if "Activity" in block.get("name", ""):
                for exercise_tuple in block.get("exercises", []):
                    details = exercise_tuple[0]

                    specific_shots_in_exercise = details.get("shots", {}).get("specific", [])
                    all_specific_shots.update(specific_shots_in_exercise)

                    tactical_shots = details.get("tactical_shots", [])
                    foundational_shots = details.get("foundational_shots", [])

                    for shot in specific_shots_in_exercise:
                        score = 0.0
                        # Tier 1: Highest score for being a 'tactical_shot'
                        if shot in tactical_shots:
                            score = 2.0
                        # Tier 2: High score for being a 'foundational_shot'
                        elif shot in foundational_shots:
                            score = 1.0

                        # --- TIER 3 LOGIC HAS BEEN REMOVED ---

                        if score > 0:
                            shot_scores[shot] = shot_scores.get(shot, 0.0) + score

        # --- Dynamic Threshold Classification ---
        primary_shots = set()
        if shot_scores:
            sorted_by_score = sorted(shot_scores.items(), key=lambda item: item[1], reverse=True)

            if len(sorted_by_score) >= top_n:
                threshold_score = sorted_by_score[top_n - 1][1]
            else:
                threshold_score = sorted_by_score[-1][1] if sorted_by_score else 0

            primary_shots = {shot for shot, score in sorted_by_score if score >= threshold_score and score > 0}

        secondary_shots = all_specific_shots - primary_shots

        return {
            "primary": sorted(list(primary_shots)),
            "secondary": sorted(list(secondary_shots)),
        }