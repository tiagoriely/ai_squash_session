# src/grammar_tools/analysis/metadata_extractor.py
from typing import Any, Dict, List
from collections import Counter
from .shot_analyser import ShotAnalyser
from .sequence_analyser import SequenceAnalyser


# Centralised thresholds for easy tuning based on your new rules.
DIFFICULTY_THRESHOLDS = {
    "beginner_max": 4,  # Includes scores 1, 2, 3, 4
    "intermediate_max": 6,  # Includes scores 5, 6
    "advanced_max": 8,  # Includes scores 7, 8
    "pro_min": 9  # Includes scores 9, 10
}
SPIKE_COUNT = 2


class MetadataExtractor:
    def __init__(self, field_retrieval_config: Dict):
        general_shot_types = field_retrieval_config.get("GENERAL_SHOT_TYPES", {})
        self.specific_to_general_map = {}
        for general, specifics in general_shot_types.items():
            for specific in specifics:
                self.specific_to_general_map[specific] = general
        self.shot_analyser = ShotAnalyser(field_retrieval_config)
        self.sequence_analyser = SequenceAnalyser()

    def _map_specific_to_general(self, specific_shots: List[str]) -> List[str]:
        general_shots = set()
        for shot in specific_shots:
            general_category = self.specific_to_general_map.get(shot)
            if general_category:
                general_shots.add(general_category)
        return sorted(list(general_shots))

    def _extract_movement(self, exercises: List[Dict]) -> List[str]:
        all_movements = set()
        for exercise_tuple in exercises:
            exercise_details = exercise_tuple[0]
            if "warmup" in exercise_details.get("types", []):
                continue
            movements = exercise_details.get("movement", [])
            all_movements.update(movements)
        return sorted(list(all_movements))

    def _determine_participants(self, exercises: List[Dict]) -> int:
        return 2

    def _determine_session_type(self, exercises):

        # archetype = self.session_plan.get("meta", {}).get("archetype", "")
        # if "conditioned_game" in archetype.lower():
        #     return "conditioned_game"
        # if "drill" in archetype.lower():
        #     return "drill"

        # Ignore warmups when deciding the session_type
        activity_exercises = [ex for ex in exercises if "warmup" not in ex[0].get("types", [])]
        if not activity_exercises:
            return "warmup_only"

        kinds = set()
        for ex in activity_exercises:
            mode = ex[2]  # tuple: (variant, value, mode, est_duration)
            if mode == "timed":
                kinds.add("drill")
            elif mode == "points":
                kinds.add("conditioned_game")

        # Fallback (shouldn't happen): use variant types if mode was missing
        if not kinds:
            for ex in activity_exercises:
                kinds.update(t for t in ex[0].get("types", []) if t in ("drill", "conditioned_game"))

        return next(iter(kinds)) if len(kinds) == 1 else f"mix({', '.join(sorted(kinds))})"

    def _calculate_session_difficulty(self, exercises: List[Dict]) -> Dict[str, Any]:
        """
        Calculates a full difficulty profile based on your new thresholds and spike rules.
        """
        scores = [ex[0].get("difficulty_score") for ex in exercises if
                  "warmup" not in ex[0].get("types", []) and ex[0].get("difficulty_score") is not None]
        if not scores:
            return {"applicable": ["intermediate", "advanced", "professional"], "recommended": "intermediate"}

        average_score = sum(scores) / len(scores)

        # 1. Determine the Base Level using the new thresholds
        base_level = "professional"
        if average_score <= DIFFICULTY_THRESHOLDS["beginner_max"]:
            base_level = "beginner"
        elif average_score <= DIFFICULTY_THRESHOLDS["intermediate_max"]:
            base_level = "intermediate"
        elif average_score <= DIFFICULTY_THRESHOLDS["advanced_max"]:
            base_level = "advanced"

        # 2. Determine Applicable Levels
        levels = ["beginner", "intermediate", "advanced", "professional"]
        applicable_levels = levels[levels.index(base_level):]

        # 3. Apply your new "Difficulty Spike" rules
        recommended_level = base_level
        if base_level == "beginner" and sum(s >= 5 for s in scores) >= SPIKE_COUNT:
            recommended_level = "intermediate"
        elif base_level == "intermediate" and sum(s >= 7 for s in scores) >= SPIKE_COUNT:
            recommended_level = "advanced"
        elif base_level == "advanced" and sum(s >= 9 for s in scores) >= SPIKE_COUNT:
            recommended_level = "professional"

        return {"applicable": applicable_levels, "recommended": recommended_level}

    def generate_rag_metadata(self, session_plan: Dict[str, Any]) -> Dict[str, Any]:
        base_meta = session_plan.get("meta", {})
        all_exercises = [ex for block in session_plan["blocks"] for ex in block.get("exercises", [])]

        # calling analysers
        classified_shots = self.shot_analyser.classify_shots(session_plan)
        exercise_sequences = self.sequence_analyser.extract_sequences(all_exercises)

        primary_shots = classified_shots.get("primary", [])
        secondary_shots = classified_shots.get("secondary", [])
        all_specific_shots = sorted(primary_shots + secondary_shots)
        general_shots = self._map_specific_to_general(all_specific_shots)
        difficulty_profile = self._calculate_session_difficulty(all_exercises)

        side_value = base_meta.get("context", {}).get("must_use_side", ["forehand", "backhand"])
        shot_side_final = [side_value] if isinstance(side_value, str) else side_value

        final_meta = {
            "session_type": self._determine_session_type(all_exercises),
            "archetype": base_meta.get("archetype"),
            "structure_id": base_meta.get("structure_id"),
            "duration": base_meta.get("duration"),
            "shotSide": shot_side_final,
            "participants": self._determine_participants(all_exercises),
            "applicable_squash_levels": difficulty_profile["applicable"],
            "recommended_squash_level": difficulty_profile["recommended"],
            "shots_general": general_shots,
            "shots_specific_primary": primary_shots,
            "shots_specific_secondary": secondary_shots,
            "movement": self._extract_movement(all_exercises),
            "rest_minutes": base_meta.get("rest_minutes", 1.5),
            "exercise_sequences": exercise_sequences
        }
        return final_meta