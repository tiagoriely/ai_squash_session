# ────────────────────────────────
# File: src/grammar_tools/planner.py
# ────────────────────────────────
from __future__ import annotations

import datetime
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


class Planner:
    """
    Reads an experiment configuration, loads grammar components,
    assembles a session plan, and saves the output.
    """

    # ---------- helper: robust action count ----------
    @staticmethod
    def _get_allowed_actions(variant):
        """
        Return the list of allowed actions no matter where it is stored.

        1. variant["allowed_actions"]
        2. variant["rules"]["allowed_actions"]
        """
        if variant.get("allowed_actions") is not None:
            return variant["allowed_actions"]
        return variant.get("rules", {}).get("allowed_actions", [])
    # -------------------------------------------------

    def __init__(self, grammar_path: str = "grammar", experiments_path: str = "experiments") -> None:
        self.grammar_path = Path(grammar_path)
        self.experiments_path = Path(experiments_path)
        self.squash_grammar_path = self.grammar_path / "sports" / "squash"
        print(f"Planner initialized. Using squash grammar at: {self.squash_grammar_path}")

    # ─────────── helper loaders ───────────
    def load_yaml(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except FileNotFoundError:
            print(f"FATAL ERROR: '{file_path}' not found.")
            sys.exit(1)
        except yaml.YAMLError as err:
            print(f"FATAL ERROR: YAML syntax problem in '{file_path}'. Error: {err}")
            sys.exit(1)

    @staticmethod
    def _resolve_duration(duration_obj):
        """Accepts int, float or {'target': X} and always returns a number."""
        if isinstance(duration_obj, dict):
            return duration_obj.get("target", 0)
        return duration_obj if isinstance(duration_obj, (int, float)) else 0

    @staticmethod
    def _coerce_variants_list(warmups_doc) -> list:
        """
        Normalizes warmups YAML to a list of variants regardless of layout.
        Accepts:
          • [ {variant}, ... ]
          • { variants: [ ... ] }
          • { content:  [ ... ] }
        """
        if warmups_doc is None:
            return []
        if isinstance(warmups_doc, list):
            return warmups_doc
        if isinstance(warmups_doc, dict):
            return warmups_doc.get("variants") or warmups_doc.get("content") or []
        return []

    # ─────────── warm-up builder ───────────
    def _process_warmup_block(
        self,
        block_tmpl: dict,
        warmup_variants: list,
        selection_config: dict | None = None,
    ) -> dict:
        """
        Build a warm-up block.

        • If `selection_config` contains `variant_id_to_use`, pick that exact warm-up.
        • Else pick the first compound warm-up.
        • Else fall back to a random simple warm-up.
        """
        block = {"block_name": block_tmpl["block_name"], "exercises": []}

        # --- choose warm-up variant -----------------------------------------
        chosen = None
        if selection_config and selection_config.get("variant_id_to_use"):
            vid = selection_config["variant_id_to_use"]
            print(f"  -> Warm-up selector: using variant_id '{vid}'")
            chosen = next((v for v in warmup_variants if v.get("variant_id") == vid), None)

        elif selection_config and selection_config.get("random"):
            print("  -> Warm-up selector: random=True → sampling any warm-up")
            if not warmup_variants:
                raise ValueError("No warm-up variants available.")
            chosen = random.choice(warmup_variants)

        if chosen is None:
            chosen = next((v for v in warmup_variants if isinstance(v, dict) and "composition" in v), None)

        if chosen is None:
            if not warmup_variants:
                raise ValueError("No warm-up variants available.")
            chosen = random.choice(warmup_variants)
        # --------------------------------------------------------------------

        # --- compound warm-up ----------------------------------------------
        if isinstance(chosen, dict) and "composition" in chosen:
            print(f"  -> Using compound warm-up: '{chosen.get('name')}'")
            components = chosen["composition"].get("components", [])

            total = sum(self._resolve_duration(e["duration_minutes"])
                        for e in block_tmpl["exercises"])

            # handle optional role switch
            if chosen["composition"].get("repeat_with_role_switch"):
                half = total / 2
                for role_tag in ("Role A", "Role B"):
                    for comp in components:
                        block["exercises"].append(
                            {
                                "name": f"{comp['name']} ({role_tag})",
                                "duration_minutes": half * comp.get("proportion", 0),
                            }
                        )
            else:
                for comp in components:
                    block["exercises"].append(
                        {
                            "name": comp.get("name", "Unnamed Component"),
                            "duration_minutes": total * comp.get("proportion", 0),
                        }
                    )

        # --- simple warm-up -------------------------------------------------
        else:
            for tmpl_slot in block_tmpl["exercises"]:
                block["exercises"].append(
                    {
                        "name": (chosen.get("name") if isinstance(chosen, dict) else str(chosen)) or "Unnamed Warm-up",
                        "duration_minutes": self._resolve_duration(tmpl_slot["duration_minutes"]),
                    }
                )

        return block

    # ─────────── master orchestrator ───────────
    def generate_plan(self, config_path: str):
        print(f"\nLoading experiment config from: {config_path}")
        cfg = self.load_yaml(config_path)
        inputs = cfg.get("inputs", {})

        structure = self.load_yaml(
            self.squash_grammar_path / "session_structures" / f"{inputs['session_structure_id']}.yaml"
        )
        warmups_doc = self.load_yaml(self.squash_grammar_path / "exercises" / "warmups.yaml")
        warmup_variants = self._coerce_variants_list(warmups_doc)

        family_file = inputs["target_family_id"].split(".")[-1] + ".yaml"
        exercise_family = self.load_yaml(self.squash_grammar_path / "exercises" / family_file)

        print("Grammar components loaded. Building session…")
        plan = {"title": exercise_family.get("family", "Session"), "blocks": []}

        # ───── progressive family archetype ─────
        variants = exercise_family.get("variants", [])
        sorted_variants = sorted(
            variants,
            key=lambda v: len(self._get_allowed_actions(v))
            if self._get_allowed_actions(v)
            else float("inf"),
        )

        activity_blocks = [b for b in structure["blocks"] if "Activity" in b["block_name"]]
        progression = sorted_variants[: len(activity_blocks)]
        prog_idx = 0

        warmup_selection = inputs.get("warmup_selection")

        for block_tmpl in structure["blocks"]:
            name = block_tmpl["block_name"]

            if "Warm-up" in name:
                block = self._process_warmup_block(block_tmpl, warmup_variants, warmup_selection)

            elif "Activity" in name:
                block = {"block_name": name, "exercises": []}
                if prog_idx < len(progression):
                    variant = progression[prog_idx]
                    family_display = exercise_family.get("family", "Family")
                    sides = next(
                        (p["options"] for p in exercise_family.get("parameters", [])
                         if p["name"] == "shotSide"),
                        [],
                    )
                    for slot, side in zip(block_tmpl["exercises"], sides):
                        block["exercises"].append(
                            {
                                "name": f"{family_display} {variant['name'].title()} ({side})",
                                "duration_minutes": self._resolve_duration(slot["duration_minutes"]),
                            }
                        )
                    prog_idx += 1
            else:
                block = block_tmpl  # passthrough

            plan["blocks"].append(block)

        # ───── write to disk ─────
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = cfg.get("experiment_name", "unnamed_experiment")
        exp_dir = self.experiments_path / f"{timestamp}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        out_path = exp_dir / f"{timestamp}_{exp_name}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "experiment_details": {
                        "name": exp_name,
                        "timestamp": timestamp,
                        "config_file_used": str(config_path),
                    },
                    "session_plan": plan,
                },
                fh,
                indent=2,
            )

        print(f"\n✅ Success! Session saved to: {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# Compatibility shim for grammar_generator.py
# Restores: build_session, WARMUP_MIN, REST_MIN, POINTS_PER_CG
# Uses current YAML grammar to synthesize a simple warm-up + a few activities.
# ---------------------------------------------------------------------------
# Reasonable defaults used only for text rendering in grammar_generator
WARMUP_MIN: int = 10
REST_MIN: float = 1.5
POINTS_PER_CG: int = 7

def _pick_family_file(_root: Path) -> Path:
    """Return the first exercise family file (not warmups/defaults)."""
    ex_dir = _root / "exercises"
    for p in sorted(ex_dir.glob("*.yaml")):
        if p.name.lower() in {"warmups.yaml", "defaults.yaml"}:
            continue
        return p
    raise FileNotFoundError(f"No exercise family YAML found under {ex_dir}")

def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def _coerce_warmup_variants(doc) -> list:
    if doc is None:
        return []
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        return doc.get("variants") or doc.get("content") or []
    return []

def build_session(total_min: int = 60):
    """
    Return (warmup_activity, blocks) where:
      warmup_activity is a loader.Activity
      blocks is List[Tuple[loader.Activity, int_points]]
    This is intentionally lightweight—just enough for grammar_generator.
    """
    from .loader import Activity  # lazy import to avoid cycles

    squash_root = Path("grammar") / "sports" / "squash"

    # Warm-up (robust to list/dict layouts)
    warmups_doc = _load_yaml(squash_root / "exercises" / "warmups.yaml")
    warm_variants = _coerce_warmup_variants(warmups_doc)
    if not warm_variants:
        warm_name = "General Movement Warm-up"
        warm_id = "warmup.default"
    else:
        w0 = warm_variants[0]
        warm_name = w0.get("name", "Warm-up") if isinstance(w0, dict) else str(w0)
        warm_id = f"warmup.{(w0.get('variant_id') if isinstance(w0, dict) else 'v0')}"
    warmup_act = Activity.model_validate({
        "activity_id": warm_id,
        "name": warm_name,
        "rules": {"sequence": "movement and light hitting"}  # cosmetic for renderer
    })

    # Family & variants
    fam_path = _pick_family_file(squash_root)
    fam_doc = _load_yaml(fam_path) or {}
    fam_name = fam_doc.get("family", fam_path.stem.replace("_", " ").title())
    variants = fam_doc.get("variants", [])
    if not variants:
        raise ValueError(f"No 'variants' in {fam_path}")

    # Use first few variants (3–4) so the output is non-trivial
    k = min(4, len(variants))
    chosen = variants[:k]

    blocks: List[Tuple[Activity, int]] = []
    for v in chosen:
        vid = v.get("variant_id", "v")
        vname = v.get("name", "Variant").title()
        seq = None
        rules = v.get("rules", {})
        if isinstance(rules, dict):
            seq = rules.get("sequence")
        act = Activity.model_validate({
            "activity_id": f"{fam_name.lower().replace(' ', '_')}.{vid}",
            "name": f"{fam_name} — {vname}",
            "rules": ({"sequence": seq} if seq else None),
        })
        blocks.append((act, POINTS_PER_CG))

    return warmup_act, blocks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/grammar_tools/planner.py <path_to_config_file>")
        sys.exit(1)
    Planner().generate_plan(sys.argv[1])
