# src/grammar_tools/planner.py
from __future__ import annotations
import random
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from lark import Lark, UnexpectedToken
except ImportError:
    # This allows the code to still run if lark is not installed,
    # though validation will be disabled.
    Lark, UnexpectedToken = None, None

from src.grammar_tools.analysis.metadata_extractor import MetadataExtractor


class Planner:
    """
    Builds a session plan by unifying a high-level Archetype (strategy) with
    low-level Block Type fillers (tactics). This version includes logic for
    enforcing point progression within blocks.
    """

    def __init__(self, exercises: Dict, structures: Dict, session_types: Dict, warmups: List,
                 block_types: List, archetypes: Dict, field_retrieval_config: Dict, config: Dict):

        self.config = config

        # Get thresholds and progression rules from the main config
        self.plan_dedup_threshold = self.config.get('plan_deduplication_threshold', 0.99)
        self.enforce_plan_points_progression = self.config.get('enforce_plan_points_progression', False)

        # Keep a history of the unique plans we've created in this run.
        self.plan_history = []

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

        # Load and compile the exercise constraint grammar
        self.constraint_parser = None
        if Lark:  # Check if the import was successful
            try:
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                peg_file_path = project_root / "grammar" / "dsl" / "exercises_rule_constraints.peg"

                with open(peg_file_path, "r") as f:
                    # Instantiate the Lark parser with your PEG grammar
                    self.constraint_parser = Lark(f.read(), start='constraint')
                print(f"✅ Exercise constraint grammar loaded successfully with Lark from: {peg_file_path}")

            except FileNotFoundError:
                print(
                    f"🛑 ERROR: PEG file not found at expected path: {peg_file_path}. Constraint validation is disabled.")
            except Exception as e:
                print(f"🛑 ERROR: Failed to compile PEG with Lark: {e}. Constraint validation is disabled.")
        else:
            print(
                "⚠️ Warning: 'lark' library not found. Constraint validation is disabled. Please run 'pip install lark-parser'.")

        # After loading all data, validate the exercise library
        if self.constraint_parser:
            self._validate_exercise_library()

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

    def _get_exercise_type(self, variant: Dict, context: Dict) -> Optional[str]:
        """
        Intelligently chooses an exercise type based on the archetype's constraints.
        """
        required_type = context.get('must_use_exercise_type')
        available_types = variant.get('types', ['drill', 'conditioned_game'])

        if required_type:
            return required_type if required_type in available_types else None

        return random.choice(available_types)

    def plan_session(self) -> Dict[str, Any] | None:
        archetype_id, archetype_def = random.choice(list(self.archetypes.items()))
        print(f"  -> Building session with STRATEGY: '{archetype_def['archetype_name']}'")

        context = self._create_context_from_archetype(archetype_def)
        if context.get("prevent_variant_repetition"):
            context["used_variants"] = set()

        # Intelligent Structure Selection
        # Get the preferred structure category from the archetype, default to 'mix'
        structure_cat = archetype_def.get('structure_category', 'mix')

        # Check if the category exists in our loaded structures
        if structure_cat not in self.structures or not self.structures[structure_cat]:
            print(f"  🛑 SESSION FAILED: No structures found for category '{structure_cat}'. Discarding.")
            return None

        # Choose a structure from the correct category
        structure_pool = self.structures[structure_cat]
        structure_id, structure = random.choice(list(structure_pool.items()))

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

        # --- VALIDATION STEPS  ---
        all_exercises = [ex for block in session_plan["blocks"] for ex in block.get("exercises", [])]
        if not all_exercises:
            print(f"  🛑 SESSION FAILED VALIDATION: No exercises were planned.")
            return None

        # 1. Duration validation
        total_exercise_duration = sum(ex[3] for ex in all_exercises)
        num_activity_blocks = len(
            [b for b in session_plan["blocks"] if "Activity" in b.get("name", "") and b.get("exercises")])
        total_rest_duration = num_activity_blocks * structure.get("default_rest_minutes", 1.5)
        actual_duration = total_exercise_duration + total_rest_duration
        target_duration = structure["target_duration_minutes"]
        duration_rules = structure.get("total_duration_rules", {})
        undershoot_limit = duration_rules.get("soft_min_undershoot", 0)
        overshoot_limit = duration_rules.get("soft_max_overshoot", 0)

        upper_bound = target_duration + overshoot_limit
        lower_bound = target_duration - undershoot_limit

        if not (lower_bound <= actual_duration <= upper_bound):
            print(
                f"  🛑 SESSION FAILED DURATION CHECK: Actual time ({actual_duration:.1f}m) outside range [{lower_bound}m - {upper_bound}m]. Discarding.")  # <-- Updated print message
            return None

        # Plan points progression validation
        if self.enforce_plan_points_progression:
            if not self._validate_plan_points_progression(session_plan):
                print(f"  🛑 SESSION FAILED POINTS PROGRESSION CHECK. Discarding.")
                return None


        # Create a placeholder meta object first for the extractor to use and final duplicate check
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

        if self._is_plan_a_near_duplicate(session_plan):
            return None

        print(
            f"  ✅ SESSION PASSED VALIDATION (Type: {final_meta.get('session_type', 'unknown')}, Actual Duration: {actual_duration:.1f}m).")
        return session_plan

    def _is_plan_a_near_duplicate(self, new_plan: Dict) -> bool:
        """
        Checks if a newly generated plan is structurally too similar to one already created.
        """
        # Create a simple "signature" of the plan based on its exercise IDs.
        # This focuses on the structure, not the text.
        try:
            new_signature = "-".join(
                ex[0]['variant_id'] for block in new_plan.get("blocks", []) for ex in block.get("exercises", [])
            )
        except (KeyError, IndexError):
            # If the plan structure is weird, treat it as unique
            return False

        if not new_signature:
            return False  # An empty plan isn't a duplicate

        for old_plan_signature in self.plan_history:
            # Use Python's built-in library to get a similarity ratio
            score = difflib.SequenceMatcher(None, new_signature, old_plan_signature).ratio()

            # The check against your config value happens here!
            if score >= self.plan_dedup_threshold:
                print(f"  🛑 DUPLICATE PLAN DETECTED (Score: {score:.2f} >= {self.plan_dedup_threshold}). Discarding.")
                return True  # It's a duplicate

        # If we get through the whole loop, it's a unique plan
        self.plan_history.append(new_signature)
        return False

    def _validate_plan_points_progression(self, session_plan: Dict) -> bool:
        """
        Checks if point-based exercises in the plan follow a non-decreasing sequence.
        Allows for sequences like [7, 7, 11, 15] and [11, 11, 11].
        """
        points_sequence = [
            ex[1] for block in session_plan.get("blocks", [])
            for ex in block.get("exercises", [])
            if ex[2] == "points"
        ]

        if not points_sequence:
            return True  # No points to check, so it's valid

        # Check for non-decreasing order
        for i in range(len(points_sequence) - 1):
            if points_sequence[i] > points_sequence[i + 1]:
                print(f"    Progression FAIL: {points_sequence[i]} > {points_sequence[i + 1]} in {points_sequence}")
                return False  # Found a decrease
        return True

    def _validate_exercise_library(self):
        """
        Parses the 'constraint_formal' field of every loaded exercise variant
        to ensure it conforms to the PEG using the Lark parser.
        """
        print("  -> Validating exercise constraint library...")
        error_count = 0
        for family_id, family in self.exercises.items():
            for variant in family.get("variants", []):
                # This handles both single string and list of strings for the formal rules
                formal_rules = variant.get("rules", {}).get("constraint_formal")
                if not formal_rules:
                    continue
                if not isinstance(formal_rules, list):
                    formal_rules = [formal_rules]

                for rule in formal_rules:
                    try:
                        self.constraint_parser.parse(rule)
                    except UnexpectedToken as e:
                        print(f"  🛑 VALIDATION ERROR in variant '{variant['variant_id']}' (family: {family_id}):")
                        print(f"     Rule: '{rule}'")
                        print(f"     Lark Error: {e}")
                        error_count += 1

        if error_count == 0:
            print("  ✅ All exercise constraints are syntactically valid.")
        else:
            print(f"  🛑 Found {error_count} invalid constraint(s). Please fix before generating.")
            # Optionally, you could raise an exception to halt the program
            raise ValueError(f"Found {error_count} invalid exercise constraints.")

    # --- All other private methods (_create_context_from_archetype, _fill_tactic, etc.) remain unchanged ---

    def _create_context_from_archetype(self, archetype_def: Dict) -> Dict:
        # ... (implementation unchanged)
        context = {}
        all_family_ids = list(set(v['family_id'] for v in self.all_variants if v.get('family_id')))
        if not all_family_ids: all_family_ids = [""]

        for constraint in archetype_def.get('hard_constraints', []):
            constraint_type = constraint.get('type')
            if constraint_type == 'require_single_shotside':
                context['must_use_side'] = random.choice(["forehand", "backhand"])
            elif constraint_type == 'focus_on_single_family':
                context['must_use_family_id'] = random.choice(all_family_ids)
            elif constraint_type == 'prevent_variant_repetition':
                context['prevent_variant_repetition'] = True
            elif constraint_type == 'prefer_complexity_progression':
                context['enforce_complexity_progression'] = True
            elif constraint_type == 'require_session_type':
                context['must_use_exercise_type'] = constraint.get('value')
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
            mirrored_exercise["shotSide"] = [mirror_side]
            mirrored_exercises.append((mirrored_exercise, value, mode, duration))
        return [first_block, {"exercises": mirrored_exercises}]

    def _get_candidate_variants(self, context: Dict, side: Optional[str] = None) -> List[Dict]:
        candidate_variants = self.all_variants

        required_type = context.get('must_use_exercise_type')
        if required_type:
            # Filter variants to only those that support the required type
            candidate_variants = [
                v for v in candidate_variants
                if required_type in v.get('types', ['drill', 'conditioned_game'])
            ]

        if context.get('must_use_family_id'):
            candidate_variants = [v for v in candidate_variants if v.get('family_id') == context['must_use_family_id']]

        if side:
            def _variant_supports_side(variant, target_side):
                # 1. Check for high-constraint `shotSide` key directly on the variant.
                if target_side in variant.get("shotSide", []):
                    return True
                # 2. Fallback to check balanced `family_parameters` for compatibility.
                if any(p.get('name') == 'shotSide' and target_side in p.get('options', []) for p in
                       variant.get('family_parameters', [])):
                    return True
                return False

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

        exercise_type = self._get_exercise_type(variant, context)
        if not exercise_type: return None

        goal = self._get_random_goal(exercise_type)
        used_variants = context.get("used_variants")
        if used_variants is not None:
            used_variants.add((variant['variant_id'], 'forehand'));
            used_variants.add((variant['variant_id'], 'backhand'))
        instance_fh, instance_bh = dict(variant), dict(variant)
        instance_fh['shotSide'] = ['forehand']
        instance_bh['shotSide'] = ['backhand']
        return [self._package_exercise(instance_fh, goal, exercise_type),
                self._package_exercise(instance_bh, goal, exercise_type)]

    def _fill_same_side_variant_progression(self, rules: Dict, context: Dict) -> List | None:
        side = context.get('must_use_side') or random.choice(["forehand", "backhand"])
        candidate_variants = self._get_candidate_variants(context, side=side)
        if len(candidate_variants) < 2: return None
        v1, v2 = random.sample(candidate_variants, k=2)
        if len(self._get_allowed_actions(v1)) > len(self._get_allowed_actions(v2)): v1, v2 = v2, v1

        type1 = self._get_exercise_type(v1, context)
        type2 = self._get_exercise_type(v2, context)
        if not type1 or not type2: return None

        goal1 = self._get_random_goal(type1);
        goal2 = self._get_random_goal(type2)

        if rules.get('enforce_points_progression') and type1 == 'conditioned_game' and type2 == 'conditioned_game':
            if goal1 > goal2: goal1, goal2 = goal2, goal1

        used_variants = context.get("used_variants")
        if used_variants is not None:
            used_variants.add((v1['variant_id'], side));
            used_variants.add((v2['variant_id'], side))

        instance1, instance2 = dict(v1), dict(v2)
        instance1['shotSide'] = [side]
        instance2['shotSide'] = [side]
        return [self._package_exercise(instance1, goal1, type1), self._package_exercise(instance2, goal2, type2)]

    def _fill_from_drills_to_condition(self, rules: Dict, context: Dict) -> List | None:
        side = context.get('must_use_side') or random.choice(["forehand", "backhand"])
        candidate_variants = self._get_candidate_variants(context, side=side)
        if not context.get('must_use_family_id'): return None

        # This block type is inherently mixed, so the 'require_session_type' constraint cannot apply.
        # Added a check to ensure we don't try to fill it for a drill-only or game-only session.
        if context.get('must_use_exercise_type'):
            return None

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
                    drill_instance['shotSide'] = [side]
                    cg_instance['shotSide'] = [side]
                    goal_drill = self._get_random_goal('drill');
                    goal_cg = self._get_random_goal('conditioned_game')
                    return [self._package_exercise(drill_instance, goal_drill, 'drill'),
                            self._package_exercise(cg_instance, goal_cg, 'conditioned_game')]
        return None

    def _fill_cross_family_action_similarity(self, rules: Dict, context: Dict) -> List | None:
        if 'must_use_family_id' in context: return None
        threshold = rules.get('require_action_similarity_threshold', 0.35)

        sequence_type = self._get_exercise_type({"types": rules['allowed_sequences']}, context)
        if not sequence_type: return None

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
                    instance1['shotSide'] = [side]
                    instance2['shotSide'] = [side]
                    return [self._package_exercise(instance1, goal1, sequence_type),
                            self._package_exercise(instance2, goal2, sequence_type)]
        return None

    @staticmethod
    def _calculate_action_similarity(actions1: List[str], actions2: List[str]) -> float:
        set1, set2 = set(actions1), set(actions2)
        intersection, union = len(set1.intersection(set2)), len(set1.union(set2))
        return intersection / union if union > 0 else 0.0