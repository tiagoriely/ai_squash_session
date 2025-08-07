import yaml
import json
import datetime
from pathlib import Path
import sys
import random


class Planner:
    """
    Reads an experiment configuration, loads grammar components,
    assembles a session plan, and saves the output.
    """

    def __init__(self, grammar_path="grammar", experiments_path="experiments"):
        self.grammar_path = Path(grammar_path)
        self.experiments_path = Path(experiments_path)
        self.squash_grammar_path = self.grammar_path / 'sports' / 'squash'
        print(f"Planner initialized. Using squash grammar at: '{self.squash_grammar_path}'")

    def load_yaml(self, file_path):
        """Helper to load a YAML file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: The file '{file_path}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: Could not parse the YAML file '{file_path}'. Error: {e}")
            sys.exit(1)

    def _resolve_duration(self, duration_obj):
        """Resolves a duration object to a single integer."""
        if isinstance(duration_obj, dict):
            return duration_obj.get('target', 0)
        return duration_obj if isinstance(duration_obj, (int, float)) else 0

    def generate_plan(self, config_path):
        """Main method to generate a plan from a config file."""
        print(f"\nLoading experiment config from: {config_path}")
        config = self.load_yaml(config_path)

        archetype_id = config.get('archetype_to_use')
        inputs = config.get('inputs', {})
        if not archetype_id or not inputs:
            print("FATAL ERROR: Config file is missing 'archetype_to_use' or 'inputs' keys.")
            sys.exit(1)

        archetype = self.load_yaml(self.squash_grammar_path / 'session_archetypes' / f"{archetype_id}.yaml")
        structure = self.load_yaml(
            self.squash_grammar_path / 'session_structures' / f"{inputs.get('session_structure_id')}.yaml")
        print("Grammar components loaded successfully.")

        print("Assembling session plan...")
        final_plan = {
            "title": archetype.get('archetype_name', 'Generated Session'),
            "target_duration_minutes": structure.get('target_duration_minutes'),
            "blocks": []
        }

        if archetype_id == 'progressive_family':
            warmup_data = self.load_yaml(self.squash_grammar_path / 'exercises' / 'warmups.yaml')
            family_id = inputs.get('target_family_id')
            family_filename = family_id.split('.')[-1] + '.yaml'
            exercise_family = self.load_yaml(self.squash_grammar_path / 'exercises' / family_filename)

            # --- THIS IS THE NEW LINE TO UPDATE THE TITLE ---
            final_plan['title'] = exercise_family.get('family', archetype.get('archetype_name'))

            progression = inputs.get('progression_order', [])

            available_warmups = warmup_data if isinstance(warmup_data, list) else warmup_data.get('variants', [])
            progression_idx = 0

            for block_template in structure.get('blocks', []):
                block_name = block_template.get('block_name')
                resolved_block = {"block_name": block_name, "exercises": []}
                num_slots = len(block_template.get('exercises', []))

                if block_name == 'Warm-up':
                    selected_variant = available_warmups[0] if available_warmups else None
                    if selected_variant and 'composition' in selected_variant:
                        components = selected_variant['composition'].get('components', [])
                        total_duration = sum(
                            self._resolve_duration(slot['duration_minutes']) for slot in block_template['exercises'])

                        for component in components:
                            resolved_block['exercises'].append({
                                "name": component.get('name'),
                                "duration_minutes": total_duration * component.get('proportion', 0)
                            })
                else:  # Handle Activity Blocks
                    if progression_idx < len(progression):
                        variant_id = progression[progression_idx]
                        side_param = next((p for p in exercise_family.get('parameters', []) if p['name'] == 'shotSide'),
                                          None)

                        if side_param and len(side_param.get('options', [])) == num_slots:
                            sides = side_param['options']
                            for i in range(num_slots):
                                resolved_block['exercises'].append({
                                    "name": f"{variant_id.replace('_', ' ').title()} ({sides[i]})",
                                    "duration_minutes": self._resolve_duration(
                                        block_template['exercises'][i]['duration_minutes'])
                                })
                        progression_idx += 1

                final_plan['blocks'].append(resolved_block)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get('experiment_name', 'unnamed_experiment')
        exp_dir = self.experiments_path / f"{timestamp}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created experiment directory: {exp_dir}")

        output_data = {
            "experiment_details": {"name": exp_name, "timestamp": timestamp, "config_file_used": str(config_path)},
            "session_plan": final_plan
        }

        output_filename = f"{timestamp}_{exp_name}.json"
        output_path = exp_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Success! Generated session plan and saved to: {output_path}")
        return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/grammar_tools/planner.py <path_to_config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    planner = Planner()
    planner.generate_plan(config_file_path)