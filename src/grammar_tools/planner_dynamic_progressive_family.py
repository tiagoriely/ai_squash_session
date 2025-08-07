import yaml
import json
import datetime
from pathlib import Path
import sys
import random


class Planner:
    def __init__(self, grammar_path="grammar", experiments_path="experiments"):
        self.grammar_path = Path(grammar_path)
        self.experiments_path = Path(experiments_path)
        self.squash_grammar_path = self.grammar_path / 'sports' / 'squash'
        print(f"Planner initialized. Using squash grammar at: '{self.squash_grammar_path}'")

    def load_yaml(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: The file '{file_path}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: Could not parse '{file_path}'. Please check its syntax. Error: {e}")
            sys.exit(1)

    def _resolve_duration(self, duration_obj):
        if isinstance(duration_obj, dict):
            return duration_obj.get('target', 0)
        return duration_obj if isinstance(duration_obj, (int, float)) else 0

    def generate_plan(self, config_path):
        print(f"\nLoading experiment config from: {config_path}")
        config = self.load_yaml(config_path)

        archetype_id = config.get('archetype_to_use')
        inputs = config.get('inputs', {})

        archetype = self.load_yaml(self.squash_grammar_path / 'session_archetypes' / f"{archetype_id}.yaml")
        structure = self.load_yaml(
            self.squash_grammar_path / 'session_structures' / f"{inputs.get('session_structure_id')}.yaml")

        warmup_data = self.load_yaml(self.squash_grammar_path / 'exercises' / 'warmups.yaml')
        family_id = inputs.get('target_family_id')
        family_filename = family_id.split('.')[-1] + '.yaml'
        exercise_family = self.load_yaml(self.squash_grammar_path / 'exercises' / family_filename)

        print("Grammar components loaded successfully.")
        print("Assembling session plan...")

        final_plan = {
            "title": exercise_family.get('family', 'Generated Session'),
            "blocks": []
        }

        if archetype_id == 'progressive_family':
            # --- REWRITTEN LOGIC FOR ROBUSTNESS ---
            all_variants = exercise_family.get('variants', [])
            sorted_variants = sorted(all_variants, key=lambda v: len(v.get('allowed_actions', [])))

            activity_block_templates = [b for b in structure.get('blocks', []) if
                                        'Activity Block' in b.get('block_name', '')]
            progression = sorted_variants[:len(activity_block_templates)]
            progression_idx = 0

            for block_template in structure.get('blocks', []):
                block_name = block_template.get('block_name')
                print(f"\nProcessing block: '{block_name}'")
                resolved_block = {"block_name": block_name, "exercises": []}

                if 'Warm-up' in block_name:
                    selected_warmup = warmup_data[0] if warmup_data else None
                    if selected_warmup and 'composition' in selected_warmup:
                        print("  -> Found compound warm-up. Deconstructing...")
                        total_duration = sum(
                            self._resolve_duration(s['duration_minutes']) for s in block_template['exercises'])
                        for component in selected_warmup['composition'].get('components', []):
                            resolved_block['exercises'].append({
                                "name": component.get('name'),
                                "duration_minutes": total_duration * component.get('proportion', 0)
                            })
                elif 'Activity Block' in block_name:
                    if progression_idx < len(progression):
                        variant = progression[progression_idx]
                        print(f"  -> Filling with progression step: '{variant.get('variant_id')}'")
                        side_param = next((p for p in exercise_family.get('parameters', []) if p['name'] == 'shotSide'),
                                          None)

                        num_slots = len(block_template.get('exercises', []))
                        num_options = len(side_param.get('options', [])) if side_param else 0

                        print(
                            f"  -> Logic check: Found 'shotSide' parameter? {'Yes' if side_param else 'No'}. Num options: {num_options}. Num slots: {num_slots}.")

                        if side_param and num_options == num_slots:
                            print("  -> Condition met. Splitting by Forehand/Backhand.")
                            sides = side_param['options']
                            for i in range(num_slots):
                                resolved_block['exercises'].append({
                                    "name": f"{variant.get('name', '').title()} ({sides[i]})",
                                    "duration_minutes": self._resolve_duration(
                                        block_template['exercises'][i]['duration_minutes'])
                                })
                        else:
                            print("  -> Condition failed. Using fallback naming.")
                        progression_idx += 1

                final_plan['blocks'].append(resolved_block)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get('experiment_name', 'unnamed_experiment')
        exp_dir = self.experiments_path / f"{timestamp}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        output_data = {
            "experiment_details": {"name": exp_name, "timestamp": timestamp, "config_file_used": str(config_path)},
            "session_plan": final_plan
        }

        output_filename = f"{timestamp}_{exp_name}.json"
        output_path = exp_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Success! Generated session plan and saved to: {output_path}")


if __name__ == '__main__':
    config_file_path = sys.argv[1]
    Planner().generate_plan(config_file_path)