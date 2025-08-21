# ───────────────────────────────────────
# File: src/grammar_tools/archetype_logic.py
# ───────────────────────────────────────
import random


# ---------- NEW HELPER ----------
def _get_allowed_actions(variant):
    """
    Return the list of allowed actions no matter where it is stored.
    Priority: variant["allowed_actions"] ➜ variant["rules"]["allowed_actions"]
    """
    if variant.get("allowed_actions") is not None:
        return variant["allowed_actions"]
    return variant.get("rules", {}).get("allowed_actions", [])
# ---------------------------------


def _resolve_duration(duration_obj):
    if isinstance(duration_obj, dict):
        return duration_obj.get("target", 0)
    return duration_obj if isinstance(duration_obj, (int, float)) else 0


# ───────── archetype entrypoint ────────
def process_progressive_family(structure, exercise_family, warmup_variants):
    print("  -> Using 'process_progressive_family'.")
    final_blocks = []

    # ---------- sort variants ----------
    variants = exercise_family.get("variants", [])
    sorted_variants = sorted(
        variants,
        key=lambda v: len(_get_allowed_actions(v))
        if _get_allowed_actions(v)
        else float("inf"),
    )

    activity_templates = [b for b in structure["blocks"] if "Activity" in b["block_name"]]
    progression = sorted_variants[: len(activity_templates)]
    prog_idx = 0
    # -----------------------------------

    for block_tmpl in structure["blocks"]:
        block_name = block_tmpl["block_name"]
        block = {"block_name": block_name, "exercises": []}

        # ----- warm-ups -----
        if "Warm-up" in block_name:
            slots = len(block_tmpl["exercises"])
            simple_warmups = [v for v in warmup_variants if "composition" not in v]
            for slot, wu in zip(block_tmpl["exercises"], random.sample(simple_warmups, slots)):
                block["exercises"].append(
                    {
                        "name": wu.get("name", "Unnamed Warm-up"),
                        "duration_minutes": _resolve_duration(slot["duration_minutes"]),
                    }
                )

        # ----- progression activities -----
        elif "Activity" in block_name:
            if prog_idx < len(progression):
                variant = progression[prog_idx]
                sides = next(
                    (p["options"] for p in exercise_family.get("parameters", [])
                     if p["name"] == "shotSide"),
                    [],
                )
                for slot, side in zip(block_tmpl["exercises"], sides):
                    block["exercises"].append(
                        {
                            "name": f"{variant['name'].title()} ({side})",
                            "duration_minutes": _resolve_duration(slot["duration_minutes"]),
                        }
                    )
                prog_idx += 1

        # ----- any other block types just copy -----
        else:
            block = block_tmpl

        final_blocks.append(block)

    return final_blocks
