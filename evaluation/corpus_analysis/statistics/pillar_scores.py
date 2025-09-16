# evaluation/corpus_analysis/statistics/pillar_scores.py
import math

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def compute_pillar_scores(diversity: dict, structure: dict, reliability: dict) -> dict:
    """
    Compute simplified, interpretable pillar scores in [0,1].

    Diversity:
      Focuses on conceptual variety (via base variant entropy) and session uniqueness.
      D = 0.7 * H_base' + 0.3 * Jaccard_base
      where H_base'     = diversity["variant_base_norm"]["entropy_norm"],
            Jaccard_base = diversity["session_jaccard"]["base_norm"]["mean"]

    Structure:
      Focuses on adherence to the declared session template/archetype. This
      is a direct measure of high-level structural integrity.
      S = template_adherence_rate
      where template_adherence_rate = structure["template_adherence_rate"]

    Reliability:
      Focuses on adherence to explicit rules and archetypes.
      R = 0.6 * rule_pass_rate + 0.4 * archetype_adherence
      where rule_pass_rate    = reliability["overall_pass_rate"],
            archetype_adherence = reliability["overall_adherence_rate"]
    """
    # --- Diversity  ---
    H_base = diversity.get("variant_base_norm", {}).get("entropy_norm", 0.0)
    jacc_b = diversity.get("session_jaccard", {}).get("base_norm", {}).get("mean", 0.0)
    D = clamp01(0.7 * H_base + 0.3 * jacc_b)

    # --- Structure  ---
    # This correctly measures adherence to the session's declared format,
    # which is the best indicator of structural integrity.
    template_adherence = structure.get("template_adherence_rate", 0.0)
    S = clamp01(template_adherence)

    # --- Reliability  ---
    rule_pass_rate = reliability.get("overall_pass_rate", 0.0)
    archetype_adherence = reliability.get("overall_adherence_rate", 0.0)
    R = clamp01(0.6 * rule_pass_rate + 0.4 * archetype_adherence)

    # --- Overall Impact  ---
    # Down-weighting Structure as it has low variance across corpora, and
    # emphasising the core trade-off between Diversity and Reliability.
    overall_impact_score = 0.4 * D + 0.2 * S + 0.4 * R

    return {"diversity_score": D, "structure_score": S, "reliability_score": R,
            "overall_impact": overall_impact_score}