# src/grammar_tools/planner.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from src.grammar_tools.analysis.metadata_extractor import MetadataExtractor


class Planner:
    """
    Builds a session plan by unifying a high-level Archetype (strategy) with
    low-level Block Type fillers (tactics). This version includes logic for
    enforcing point progression within blocks.
    """

    def __init__(self, exercises: Dict, structures: Dict, session_types: Dict, warmups: List,
                 block_types: List, archetypes: Dict, field_retrieval_config: Dict):
        self.exercises = exercises
        self.structures = structures
        self.session_types = session_types
        self.warmups = warmups
        self.block_types = block_types
        self.archetypes = archetypes
        self.all_variants = self._flatten_variants()

        # Instantiate the new helper
        self.metadata_extractor = MetadataExtractor(field_retrieval_config)

        print(f"✅ Planner initialized with support for point progression.")

    def _flatten_variants(self) -> List[Dict]:
        """Creates a flat list of all variants from all families, enriched with family info."""
        flat_list = []
        for family_id, family_data in self.exercises.items():
            for variant in family_data.get("variants", []):
                instance = dict(variant)
                instance['family_id'] = family_id
                instance['family_name'] = family_data.get('family', 'Unknown Family')
                instance['family_parameters'] = family_data.get('parameters', [])
                flat_list.append(instance)
        return flat_list

    @staticmethod
    def _get_allowed_actions(variant: Dict) -> List:
        if variant.get("allowed_actions") is not None:
            return variant["allowed_actions"]
        return variant.get("rules", {}).get("allowed_actions", [])

    def _package_exercise(self, variant_instance: Dict, goal: Any, exercise_type: str) -> Tuple[Dict, Any, str, float]:
        """
        Packages a variant instance into the final exercise tuple format.
        Now takes the 'goal' (points or duration) as an argument.
        """
        if exercise_type in ('drill', 'warmup'):
            return (variant_instance, goal, "timed", float(goal))
        else:  # conditioned_game
            point_duration = self.session_types['conditioned_game']['point_duration_minutes']
            estimated_duration = goal * point_duration
            return (variant_instance, goal, "points", estimated_duration)

    def _get_random_goal(self, exercise_type: str) -> Any:
        """Gets a random goal (duration or points) for an exercise type."""
        if exercise_type in ('drill', 'warmup'):
            return self.session_types['drill']['default_duration_minutes_by_level']['intermediate']
        else:  # conditioned_game
            points_opts = self.session_types['conditioned_game']['allowed_points_per_exercise']
            return random.choices([p['points'] for p in points_opts], [p['weight'] for p in points_opts])[0]

    def plan_session(self) -> Dict[str, Any] | None:
        archetype_id, archetype_def = random.choice(list(self.archetypes.items()))
        print(f"  -> Building session with STRATEGY: '{archetype_def['archetype_name']}'")

        context = self._create_context_from_archetype(archetype_def)
        if context.get("prevent_variant_repetition"):
            context["used_variants"] = set()

        structure_id, structure = random.choice(list(self.structures.items()))
        session_plan = {"blocks": []}

        # --- Build plan (logic unchanged) ---
        warmup_templates = [bt for bt in structure["blocks"] if "Warm-up" in bt["block_name"]]
        for block_template in warmup_templates:
            warmup_block = {"name": block_template["block_name"], "exercises": []}
            for _ in block_template["exercises"]:
                if not self.warmups: raise ValueError("Warmups list is empty.")
                goal = self._get_random_goal('warmup')
                warmup_block["exercises"].append(self._package_exercise(random.choice(self.warmups), goal, 'warmup'))
            session_plan["blocks"].append(warmup_block)

        activity_templates = [bt for bt in structure["blocks"] if "Activity" in bt["block_name"]]
        while activity_templates:
            block_template = activity_templates[0]
            print(f"  -> Attempting to fill '{block_template['block_name']}'...")
            block_filled_successfully = False
            allowed_block_ids = archetype_def.get('allowed_block_types',
                                                  [bt['block_type_id'] for bt in self.block_types])
            random.shuffle(allowed_block_ids)
            for block_type_id in allowed_block_ids:
                block_type_def = next((bt for bt in self.block_types if bt['block_type_id'] == block_type_id), None)
                if not block_type_def: continue
                num_blocks_needed = block_type_def.get("rules", {}).get("block_count", 1)
                if len(activity_templates) < num_blocks_needed: continue
                newly_filled_blocks = self._fill_tactic(block_type_def, context)
                if newly_filled_blocks:
                    print(f"    SUCCESS with TACTIC: '{block_type_id}' (filled {len(newly_filled_blocks)} block(s))")
                    for i, block in enumerate(newly_filled_blocks):
                        block["name"] = activity_templates[i]["block_name"]
                        session_plan["blocks"].append(block)
                    activity_templates = activity_templates[num_blocks_needed:]
                    block_filled_successfully = True
                    break
            if not block_filled_successfully:
                print(f"    FAILURE: Could not fill '{block_template['block_name']}'. Discarding session.")
                return None

        # --- DURATION VALIDATION STEP (logic unchanged) ---
        all_exercises = [ex for block in session_plan["blocks"] for ex in block.get("exercises", [])]
        if not all_exercises:
            print(f"  🛑 SESSION FAILED VALIDATION: No exercises were planned.")
            return None

        total_exercise_duration = sum(ex[3] for ex in all_exercises)
        num_activity_blocks = len(
            [b for b in session_plan["blocks"] if "Activity" in b.get("name", "") and b.get("exercises")])
        total_rest_duration = num_activity_blocks * structure.get("default_rest_minutes", 1.5)
        actual_duration = total_exercise_duration + total_rest_duration
        target_duration = structure["target_duration_minutes"]
        undershoot_limit = structure.get("total_duration_rules", {}).get("soft_min_undershoot", 0)

        if actual_duration > target_duration or actual_duration < (target_duration - undershoot_limit):
            print(
                f"  🛑 SESSION FAILED DURATION CHECK: Actual time ({actual_duration:.1f}m) outside range [{target_duration - undershoot_limit}m - {target_duration}m]. Discarding.")
            return None

        # --- REFACTORED METADATA GENERATION ---
        # Create a placeholder meta object first for the extractor to use
        session_plan["meta"] = {
            "archetype": archetype_def['archetype_name'],
            "structure_id": structure_id,
            "duration": target_duration,
            "rest_minutes": structure.get("default_rest_minutes", 1.5),
            "context": {k: v for k, v in context.items() if k != 'used_variants'}
        }

        # Delegate the final, rich metadata creation to the specialized class
        final_meta = self.metadata_extractor.generate_rag_metadata(session_plan)
        session_plan["meta"] = final_meta

        print(
            f"  ✅ SESSION PASSED VALIDATION (Type: {final_meta.get('session_type', 'unknown')}, Actual Duration: {actual_duration:.1f}m).")
        return session_plan

    # --- All other private methods (_create_context_from_archetype, _fill_tactic, etc.) remain unchanged ---

    def _create_context_from_archetype(self, archetype_def: Dict) -> Dict:
        # ... (implementation unchanged)
        context = {}
        all_family_ids = list(set(v['family_id'] for v in self.all_variants if v.get('family_id')))
        if not all_family_ids: all_family_ids = [""]
        for constraint in archetype_def.get('hard_constraints', []):
            if constraint['type'] == 'require_single_shotside': context['must_use_side'] = random.choice(
                ["forehand", "backhand"])
            if constraint['type'] == 'focus_on_single_family': context['must_use_family_id'] = random.choice(
                all_family_ids)
            if constraint['type'] == 'prevent_variant_repetition': context['prevent_variant_repetition'] = True
            if constraint['type'] == 'prefer_complexity_progression': context['enforce_complexity_progression'] = True
        return context

    def _fill_tactic(self, block_type_def: Dict, context: Dict) -> List[Dict] | None:
        # ... (implementation unchanged)
        tactic_id = block_type_def['block_type_id']
        if 'symmetrical' in tactic_id and 'pair' in tactic_id:
            single_tactic_id = tactic_id.replace('symmetrical_', '').replace('_pair', '')
            single_block_def = next((bt for bt in self.block_types if bt['block_type_id'] == single_tactic_id), None)
            if not single_block_def: return None
            return self._fill_symmetrical_pair(single_block_def, context)
        filler_map = {"same_variant_side_switch": self._fill_same_variant_side_switch,
                      "same_side_variant_progression": self._fill_same_side_variant_progression,
                      "from_drills_to_condition": self._fill_from_drills_to_condition,
                      "cross_family_action_similarity": self._fill_cross_family_action_similarity}
        filler_func = filler_map.get(tactic_id)
        if not filler_func: return None
        exercises = filler_func(block_type_def.get('rules', {}), context)
        return [{"exercises": exercises}] if exercises else None

    def _fill_symmetrical_pair(self, single_block_def: Dict, context: Dict) -> List[Dict] | None:
        # ... (implementation unchanged)
        if 'must_use_side' in context: return None
        starting_side = random.choice(["forehand", "backhand"])
        first_block_context = context.copy()
        first_block_context['must_use_side'] = starting_side
        first_block_list = self._fill_tactic(single_block_def, first_block_context)
        if not first_block_list: return None
        first_block = first_block_list[0]
        mirror_side = "backhand" if starting_side == "forehand" else "forehand"
        mirrored_exercises = []
        for exercise, value, mode, duration in first_block["exercises"]:
            mirrored_exercise = dict(exercise)
            mirrored_exercise["shotSide"] = mirror_side
            mirrored_exercises.append((mirrored_exercise, value, mode, duration))
        return [first_block, {"exercises": mirrored_exercises}]

    def _get_candidate_variants(self, context: Dict, side: Optional[str] = None) -> List[Dict]:
        # ... (implementation unchanged)
        candidate_variants = self.all_variants
        if context.get('must_use_family_id'):
            candidate_variants = [v for v in candidate_variants if v.get('family_id') == context['must_use_family_id']]
        if side:
            candidate_variants = [v for v in candidate_variants if any(
                p.get('name') == 'shotSide' and side in p.get('options', []) for p in v.get('family_parameters', []))]
        used_variants = context.get("used_variants")
        if used_variants is not None:
            key_func = lambda v: (v['variant_id'], side) if side else (v['variant_id'],)
            if side:
                candidate_variants = [v for v in candidate_variants if (v['variant_id'], side) not in used_variants]
            else:  # For side-switch, check both
                candidate_variants = [v for v in candidate_variants if
                                      (v['variant_id'], 'forehand') not in used_variants and (
                                          v['variant_id'], 'backhand') not in used_variants]
        return candidate_variants

    def _fill_same_variant_side_switch(self, rules: Dict, context: Dict) -> List | None:
        # ... (implementation unchanged)
        if 'must_use_side' in context: return None
        candidate_variants = self._get_candidate_variants(context)
        candidate_variants = [v for v in candidate_variants if any(
            {"forehand", "backhand"}.issubset(set(p.get("options", []))) for p in v.get('family_parameters', []))]
        if not candidate_variants: return None
        variant = random.choice(candidate_variants)
        exercise_type = random.choice(variant.get("types", ['drill']))
        goal = self._get_random_goal(exercise_type)
        used_variants = context.get("used_variants")
        if used_variants is not None:
            used_variants.add((variant['variant_id'], 'forehand'));
            used_variants.add((variant['variant_id'], 'backhand'))
        instance_fh, instance_bh = dict(variant), dict(variant)
        instance_fh['shotSide'] = 'forehand';
        instance_bh['shotSide'] = 'backhand'
        return [self._package_exercise(instance_fh, goal, exercise_type),
                self._package_exercise(instance_bh, goal, exercise_type)]

    def _fill_same_side_variant_progression(self, rules: Dict, context: Dict) -> List | None:
        side = context.get('must_use_side') or random.choice(["forehand", "backhand"])
        candidate_variants = self._get_candidate_variants(context, side=side)
        if len(candidate_variants) < 2: return None
        v1, v2 = random.sample(candidate_variants, k=2)
        if len(self._get_allowed_actions(v1)) > len(self._get_allowed_actions(v2)): v1, v2 = v2, v1

        type1 = random.choice(v1.get("types", ['drill']));
        type2 = random.choice(v2.get("types", ['drill']))
        goal1 = self._get_random_goal(type1);
        goal2 = self._get_random_goal(type2)

        if rules.get('enforce_points_progression') and type1 == 'conditioned_game' and type2 == 'conditioned_game':
            if goal1 > goal2: goal1, goal2 = goal2, goal1

        used_variants = context.get("used_variants")
        if used_variants is not None:
            used_variants.add((v1['variant_id'], side));
            used_variants.add((v2['variant_id'], side))

        instance1, instance2 = dict(v1), dict(v2)
        instance1['shotSide'] = side;
        instance2['shotSide'] = side
        return [self._package_exercise(instance1, goal1, type1), self._package_exercise(instance2, goal2, type2)]

    def _fill_from_drills_to_condition(self, rules: Dict, context: Dict) -> List | None:
        side = context.get('must_use_side') or random.choice(["forehand", "backhand"])
        candidate_variants = self._get_candidate_variants(context, side=side)
        if not context.get('must_use_family_id'): return None
        drills = [v for v in candidate_variants if 'drill' in v.get('types', [])]
        c_games = [v for v in candidate_variants if 'conditioned_game' in v.get('types', [])]
        if not drills or not c_games: return None
        random.shuffle(drills);
        random.shuffle(c_games)
        for drill_v in drills:
            for cg_v in c_games:
                if drill_v['family_id'] == cg_v['family_id'] and len(self._get_allowed_actions(drill_v)) <= len(
                        self._get_allowed_actions(cg_v)):
                    used_variants = context.get("used_variants")
                    if used_variants is not None:
                        used_variants.add((drill_v['variant_id'], side));
                        used_variants.add((cg_v['variant_id'], side))
                    drill_instance, cg_instance = dict(drill_v), dict(cg_v)
                    drill_instance['shotSide'] = side;
                    cg_instance['shotSide'] = side
                    goal_drill = self._get_random_goal('drill');
                    goal_cg = self._get_random_goal('conditioned_game')
                    return [self._package_exercise(drill_instance, goal_drill, 'drill'),
                            self._package_exercise(cg_instance, goal_cg, 'conditioned_game')]
        return None

    def _fill_cross_family_action_similarity(self, rules: Dict, context: Dict) -> List | None:
        if 'must_use_family_id' in context: return None
        threshold = rules.get('require_action_similarity_threshold', 0.35)
        sequence_type = random.choice(rules['allowed_sequences'])[0]
        side = context.get('must_use_side') or random.choice(["forehand", "backhand"])
        candidate_variants = self._get_candidate_variants(context, side=side)
        candidate_variants = [v for v in candidate_variants if sequence_type in v.get('types', [])]
        if len(candidate_variants) < 2: return None
        random.shuffle(candidate_variants)
        for i in range(len(candidate_variants)):
            for j in range(i + 1, len(candidate_variants)):
                v1, v2 = candidate_variants[i], candidate_variants[j]
                if v1['family_id'] == v2['family_id']: continue
                actions1, actions2 = self._get_allowed_actions(v1), self._get_allowed_actions(v2)
                if self._calculate_action_similarity(actions1, actions2) >= threshold:
                    if len(actions1) > len(actions2): v1, v2 = v2, v1
                    goal1 = self._get_random_goal(sequence_type);
                    goal2 = self._get_random_goal(sequence_type)
                    if rules.get('enforce_points_progression') and sequence_type == 'conditioned_game':
                        if goal1 > goal2: goal1, goal2 = goal2, goal1
                    used_variants = context.get("used_variants")
                    if used_variants is not None:
                        used_variants.add((v1['variant_id'], side));
                        used_variants.add((v2['variant_id'], side))
                    instance1, instance2 = dict(v1), dict(v2)
                    instance1['shotSide'] = side;
                    instance2['shotSide'] = side
                    return [self._package_exercise(instance1, goal1, sequence_type),
                            self._package_exercise(instance2, goal2, sequence_type)]
        return None

    @staticmethod
    def _calculate_action_similarity(actions1: List[str], actions2: List[str]) -> float:
        set1, set2 = set(actions1), set(actions2)
        intersection, union = len(set1.intersection(set2)), len(set1.union(set2))
        return intersection / union if union > 0 else 0.0