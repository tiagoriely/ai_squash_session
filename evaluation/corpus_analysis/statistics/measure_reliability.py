# evaluation/corpus_analysis/statistics/measure_reliability.py
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re
from statistics import mean, pstdev
from evaluation.corpus_analysis.utils import load_corpus
import json


def check_archetype_adherence(corpus: List[Dict]) -> Dict[str, dict]:
    """
    Post-hoc checks that sessions conform to simple archetype rules.

    For each relevant archetype family we report:
      { "total": n, "success": k, "rate_percent": 100*k/n }

    Rules implemented:
      - Conditioned Games Only: all activities are 'conditioned_game'.
      - Drill Only: all activities are 'drill'.
      - Dynamic Block / mix(...): both modes appear at least once.
      - Progressive Family: exactly one family_id across activities.
      - Progressive Single ShotSide: a single side targeted (from meta.shotSide).
    """
    buckets = {
        "Conditioned Games Only": {"total": 0, "success": 0},
        "Drill Only": {"total": 0, "success": 0},
        "Dynamic Block (Mixed Modes)": {"total": 0, "success": 0},
        "Single Family": {"total": 0, "success": 0},
        "Single ShotSide": {"total": 0, "success": 0},
    }

    for s in corpus:
        meta = s.get("meta", {}) or {}
        arch = (meta.get("archetype") or "").lower()

        used = meta.get("exercises_used", []) or []

        # Families (from exercise_sequences to reflect *this* session)
        fams = set()
        for ex in (meta.get("exercise_sequences") or []):
            fid = ex.get("exercise_family_id")
            if fid and fid != "squash.family.warmup":
                fams.add(fid)

        # Modes & sides (from rendered contents bullets)
        bullet = re.compile(r"^\s*[•\-\*]\s")
        pts_pat = re.compile(r"\bpt[s]?:", re.IGNORECASE)
        min_pat = re.compile(r"\bmin:", re.IGNORECASE)
        side_pat = re.compile(r"\((forehand|backhand)\)", re.IGNORECASE)

        modes = []
        sides_seen = set()
        for raw in (s.get("contents") or "").splitlines():
            line = raw.strip()
            if not bullet.match(line):
                continue
            if "Rest:" in line or "Warmup" in line or "Warm-up" in line:
                continue
            if pts_pat.search(line):
                modes.append("conditioned_game")
            elif min_pat.search(line):
                modes.append("drill")
            m = side_pat.search(line)
            if m:
                sides_seen.add(m.group(1).lower())

        # Conditioned Games Only
        if "conditioned games only" in arch:
            buckets["Conditioned Games Only"]["total"] += 1
            ok = len(modes) > 0 and all(m == "conditioned_game" for m in modes)
            if ok:
                buckets["Conditioned Games Only"]["success"] += 1

        # Drill Only
        if "drill only" in arch:
            buckets["Drill Only"]["total"] += 1
            ok = len(modes) > 0 and all(m == "drill" for m in modes)
            if ok:
                buckets["Drill Only"]["success"] += 1

        # Dynamic / Mixed
        if "dynamic block" in arch or "mix(" in (meta.get("session_type") or ""):
            buckets["Dynamic Block (Mixed Modes)"]["total"] += 1
            ok = ("drill" in modes) and ("conditioned_game" in modes)
            if ok:
                buckets["Dynamic Block (Mixed Modes)"]["success"] += 1

        # Single Family (Progressive Family)
        if "progressive family" in arch:
            buckets["Single Family"]["total"] += 1
            if len(fams) == 1:
                buckets["Single Family"]["success"] += 1

        # Single ShotSide
        if "single shotside" in arch:
            buckets["Single ShotSide"]["total"] += 1
            sides = sides_seen if sides_seen else set(meta.get("shotSide") or [])
            if len(sides) == 1 and len(sides) > 0:
                buckets["Single ShotSide"]["success"] += 1

    # Compute rates
    for k, v in buckets.items():
        n = max(1, v["total"])
        v["rate_percent"] = 100.0 * v["success"] / n

    return buckets





def validate_corpus_rules(corpus: List[Dict]) -> Dict[str, float]:
    """
    Runs a suite of concrete validation checks across the corpus.
    Returns a dictionary of pass rates for each rule.
    """
    num_sessions = len(corpus)
    if num_sessions == 0:
        return {"overall_pass_rate": 1.0}

    passes = Counter()

    # We will check two simple but important rules.
    rules_to_check = ["duration_consistency", "warmup_before_intensity"]
    # We track how many sessions are applicable for each rule.
    applicable_sessions = Counter()

    for session in corpus:
        meta = session.get("meta", {}) or {}

        # Rule 1: Duration Consistency
        total_duration = meta.get("duration", 0)
        # The 'contents' field is more reliable for block durations
        durations_in_text = [int(m.group(1)) for m in
                             re.finditer(r'•\s*(\d+)\s*(?:min|pts):', session.get("contents", ""))]

        if total_duration > 0 and durations_in_text:
            applicable_sessions["duration_consistency"] += 1
            # Total activity time should not exceed the session duration minus rests
            # This is a soft check, but very useful.
            if sum(durations_in_text) <= total_duration:
                passes["duration_consistency"] += 1

        # Rule 2: Warm-up presence (a fundamental check)
        applicable_sessions["warmup_before_intensity"] += 1
        if "Warm-up" in session.get("contents", "") or "Warmup" in session.get("contents", ""):
            passes["warmup_before_intensity"] += 1

    pass_rates = {
        rule: (passes[rule] / applicable_sessions[rule]) if applicable_sessions[rule] > 0 else 1.0
        for rule in rules_to_check
    }
    pass_rates["overall_pass_rate"] = mean(pass_rates.values()) if pass_rates else 1.0

    return pass_rates


def analyse_shot_specificity(corpus: List[Dict]) -> Dict[str, float]:
    """
    Calculates the average number of primary specific shots per session.

    A lower number suggests a more focused and targeted session plan.
    """
    num_primary_shots = []
    for session in corpus:
        count = len(session.get("meta", {}).get("shots_specific_primary", []))
        num_primary_shots.append(count)

    if not num_primary_shots:
        return {"mean_primary_shots": 0.0, "std_primary_shots": 0.0}

    return {
        "mean_primary_shots": mean(num_primary_shots),
        "std_primary_shots": pstdev(num_primary_shots) if len(num_primary_shots) > 1 else 0.0
    }


def analyse_duration_deviation(corpus: List[Dict]) -> Dict[str, float]:
    """
    Calculates the average absolute deviation between stated and actual duration.

    A lower value indicates higher reliability in planning.
    """
    deviations = []
    for session in corpus:
        stated_duration = session.get("meta", {}).get("duration", 0)

        # Sum durations of timed components (drills and warm-ups) from the contents
        summed_duration = sum(
            int(m.group(1)) for m in re.finditer(r'•\s*(\d+)\s*min:', session.get("contents", ""))
        )

        if stated_duration > 0 and summed_duration > 0:
            deviation = abs(stated_duration - summed_duration)
            deviations.append(deviation)

    if not deviations:
        return {"mean_abs_deviation_mins": 0.0, "std_abs_deviation_mins": 0.0}

    return {
        "mean_abs_deviation_mins": mean(deviations),
        "std_abs_deviation_mins": pstdev(deviations) if len(deviations) > 1 else 0.0
    }


def analyse_reliability_metrics(corpus: List[Dict]) -> Dict[str, dict]:
    """
    Aggregate simplified and detailed reliability metrics.
    """
    # Core rule and adherence checks (for the pillar score)
    rule_validation_results = validate_corpus_rules(corpus)
    adherence_results = check_archetype_adherence(corpus)

    # --- NEW: Detailed quantitative checks (for deeper analysis) ---
    duration_deviation_stats = analyse_duration_deviation(corpus)
    shot_specificity_stats = analyse_shot_specificity(corpus)

    # Calculate overall adherence rate for the pillar score
    total_adherence_sessions = sum(v.get("total", 0) for v in adherence_results.values())
    total_adherence_success = sum(v.get("success", 0) for v in adherence_results.values())
    overall_adherence_rate = (
                total_adherence_success / total_adherence_sessions) if total_adherence_sessions > 0 else 1.0

    return {
        # --- Core metrics for pillar score ---
        "rule_validation": rule_validation_results,
        "archetype_adherence": adherence_results,
        "overall_adherence_rate": overall_adherence_rate,
        "overall_pass_rate": rule_validation_results["overall_pass_rate"],
        "duration_deviation": duration_deviation_stats,
        "shot_specificity": shot_specificity_stats,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the reliability of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("--json-indent", type=int, default=2, help="Pretty-print JSON with this indent.")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    results = analyse_reliability_metrics(corpus)
    print(json.dumps(results, indent=args.json_indent, ensure_ascii=False))





# def _get_activity_modes_and_intensities(session: Dict) -> List[Tuple[str, int]]:
#     """Helper to extract an ordered list of (mode, intensity) for activities."""
#     activities = []
#     for ex in (session.get("meta", {}) or {}).get("exercise_sequences", []) or []:
#         fam = ex.get("exercise_family_id")
#         if fam and fam != "squash.family.warmup":
#             # Default intensity to a moderate value if missing
#             intensity = ex.get("intensity", 3)
#             # Infer mode from variant ID if not explicit
#             mode = "drill"  # default
#             if "conditioned_game" in ex.get("exercise_variant_id", "") or "cg" in ex.get("exercise_variant_id", ""):
#                 mode = "conditioned_game"
#             activities.append((mode, intensity))
#     return activities

# def _extract_global_shot_vocab(corpus: Iterable[dict]) -> set:
#     """
#     Build the union of `shots_general` across sessions; used to score 'open rally'.
#     """
#     vocab = set()
#     for s in corpus:
#         for shot in (s.get("meta", {}) or {}).get("shots_general", []) or []:
#             if isinstance(shot, str) and shot.strip():
#                 vocab.add(shot.strip().lower())
#     return vocab or {"boast", "cross", "drive", "drop", "lob", "volley", "kill"}
#
# def _aoi_from_ast(ast_nodes: list, open_capacity: int) -> Tuple[float, dict]:
#     """
#     Compute Action Opportunity Index(AOI) from a sequence AST.
#
#     AOI = |A| + α * Σ_i (k_i - 1) + β * O + γ * R
#     where:
#       - A = set of terminal action names reachable in one pass,
#       - (k_i - 1) sums extra choice breadth across Choice nodes,
#       - O = #Optional nodes,
#       - R = 1 if any Repeat node present, else 0.
#
#     If 'open rally' appears with no control structure (only Action + optional Restart), assign a small AOI (1.0). If 'open rally' co-occurs with control structure (Choice/Optional/Repeat), set AOI := open_capacity.
#     Returns (aoi_value, debug_breakdown).
#     """
#     if not isinstance(ast_nodes, list) or not ast_nodes:
#         return 0.0, {"reason": "empty_ast"}
#
#     actions = set()
#     choice_extra = 0
#     optional_count = 0
#     has_repeat = False
#     saw_open = False
#
#     stack = list(ast_nodes)
#     while stack:
#         node = stack.pop()
#         ntype = node.get("type")
#         if ntype == "Action":
#             name = (node.get("name") or "").strip().lower()
#             if name:
#                 actions.add(name)
#                 if "open rally" in name:
#                     saw_open = True
#         elif ntype == "Choice":
#             opts = node.get("options", []) or []
#             choice_extra += max(0, len(opts) - 1)
#             stack.extend(opts)
#         elif ntype == "Optional":
#             optional_count += 1
#             body = node.get("body")
#             if isinstance(body, dict):
#                 stack.append(body)
#         elif ntype == "Repeat":
#             has_repeat = True
#             body = node.get("body")
#             if isinstance(body, dict):
#                 stack.append(body)
#         # 'Restart' has no body; we just ignore it structurally.
#
#     # Treat trivial "open rally -> restart" as low-information, not full capacity
#     if saw_open and not choice_extra and not optional_count and not has_repeat and len(actions) <= 1:
#         trivial_aoi = 1.0
#         return trivial_aoi, {
#             "open_rally": True, "trivial": True, "assigned_aoi": trivial_aoi
#         }
#     # Non-trivial open rally (appears with real control structure) can be high-capacity
#     if saw_open:
#         return float(open_capacity), {
#             "open_rally": True, "trivial": False, "open_capacity": open_capacity
#         }
#
#     alpha = beta = gamma = 1.0
#     aoi = float(len(actions) + alpha * choice_extra + beta * optional_count + gamma * (1 if has_repeat else 0))
#     return aoi, {
#         "open_rally": False,
#         "distinct_actions": len(actions),
#         "choice_extra": choice_extra,
#         "optional": optional_count,
#         "repeat": int(has_repeat),
#     }
#
# def _vps_from_variant_id(variant_id: str) -> Tuple[float, dict]:
#     """
#     Variant Permissiveness Score (fallback when AST is missing/vague).
#
#     VPS(variant_id) = sum_{tokens} w(token), with a small transparent lexicon.
#     """
#     if not variant_id:
#         return 0.0, {"tokens": [], "sum": 0.0}
#
#     tok = [t for t in re.split(r"[^a-zA-Z0-9]+", variant_id.lower()) if t]
#
#     # Normalise to a simple weight map (compose multiword forms)
#     weight = {
#         "any": 2.0,
#         "kills_allowed": 1.0,
#         "kills": 0.0, "allowed": 0.0,  # ignored alone
#         "extra": 1.0, "extra_drive": 1.0,
#         "counter_drop": 1.0, "counter": 0.0,
#         # restrictions
#         "deep_only": -1.0, "deep_onlyd": -1.0, "only": -1.0,
#         "volley": 0.0, "drop": 0.0, "deep": 0.0, "straight": 0.0, "lob": 0.0, "drive": 0.0,
#         "boast": 0.0, "cross": 0.0, "kill": 0.0,
#         "forehand": 0.0, "backhand": 0.0
#     }
#
#     # Collapse common bigrams into single logical tokens
#     collapsed = []
#     i = 0
#     while i < len(tok):
#         two = (tok[i] + "_" + tok[i+1]) if i+1 < len(tok) else None
#         if two in ("kills_allowed", "extra_drive", "counter_drop", "deep_only"):
#             collapsed.append(two); i += 2; continue
#         # deep_onlyD appears in your data; normalise to deep_onlyd
#         if tok[i].startswith("deep_onlyd"):
#             collapsed.append("deep_onlyd"); i += 1; continue
#         collapsed.append(tok[i]); i += 1
#
#     s = sum(weight.get(t, 0.0) for t in collapsed)
#     return float(s), {"tokens": collapsed, "sum": float(s)}
#
# def _iter_activity_entries(session: dict) -> Iterable[dict]:
#     for ex in (session.get("meta", {}) or {}).get("exercise_sequences", []) or []:
#         fam = ex.get("exercise_family_id")
#         if fam and fam != "squash.family.warmup":
#             yield ex
#
# def score_exercise_capacity(ex: dict, open_capacity: int) -> Tuple[float, str, dict]:
#     """
#     Returns (score, source, debug) for one activity exercise.
#     source ∈ {"AST", "VPS"}.
#     """
#     ast = ex.get("sequence_ast")
#     if isinstance(ast, list) and ast:
#         aoi, dbg = _aoi_from_ast(ast, open_capacity)
#         return aoi, "AST", dbg
#
#     # Vague ASTs like 'open rally -> restart' are still represented as AST;
#     # we treat those as AST (AOI) because _aoi_from_ast handles 'open rally'.
#     dsl = (ex.get("sequence_dsl") or "").strip().lower()
#     if "open rally" in dsl:
#         # No AST: treat as low-information, do not grant full capacity
#         return 1.0, "VPS", {"open_rally_from_dsl": True, "assigned_aoi": 1.0}
#
#     # Fallback to variant-id proxy
#     vps, dbg = _vps_from_variant_id(ex.get("exercise_variant_id", ""))
#     return vps, "VPS", dbg
#
# def _spearman_rank_corr(values: List[float]) -> float:
#     n = len(values)
#     if n < 2:
#         return 0.0
#     # Rank with average ranks for ties
#     sorted_idx = sorted(range(n), key=lambda i: values[i])
#     ranks = [0.0]*n
#     i = 0
#     while i < n:
#         j = i
#         while j+1 < n and values[sorted_idx[j+1]] == values[sorted_idx[i]]:
#             j += 1
#         r = (i + j) / 2.0 + 1.0
#         for k in range(i, j+1):
#             ranks[sorted_idx[k]] = r
#         i = j + 1
#     xs = list(range(1, n+1))
#     mx, my = mean(xs), mean(ranks)
#     num = sum((x-mx)*(y-my) for x, y in zip(xs, ranks))
#     denx = math.sqrt(sum((x-mx)**2 for x in xs))
#     deny = math.sqrt(sum((y-my)**2 for y in ranks))
#     return (num / (denx*deny)) if denx > 0 and deny > 0 else 0.0
#
# def analyse_session_progression(session: dict, open_capacity: int) -> dict:
#     """
#     Compute capacity scores for a session's ordered activities and summarise progression.
#
#     Returns:
#       {
#         "scores": [float,...],
#         "sources": ["AST"|"VPS", ...],
#         "adjacent_strict_increasing": float in [0,1],
#         "adjacent_non_decreasing": float in [0,1],
#         "spearman_r": float in [-1,1],
#         "first_last_delta": float,
#         "violations": int,
#         "ast_coverage": float in [0,1]
#       }
#     """
#     scores, sources = [], []
#     for ex in _iter_activity_entries(session):
#         s, src, _ = score_exercise_capacity(ex, open_capacity)
#         scores.append(float(s)); sources.append(src)
#
#     T = len(scores)
#     if T < 2:
#         return {
#             "scores": scores, "sources": sources,
#             "adjacent_strict_increasing": 1.0 if T <= 1 else 0.0,
#             "adjacent_non_decreasing": 1.0 if T <= 1 else 0.0,
#             "spearman_r": 0.0, "first_last_delta": 0.0,
#             "violations": 0,
#             "ast_coverage": 1.0 if all(x == "AST" for x in sources) else (0.0 if T == 0 else sources.count("AST") / T),
#         }
#
#     eps = 1e-6
#     inc_steps = sum(1 for a, b in zip(scores, scores[1:]) if (b - a) > eps)
#     dec_steps = sum(1 for a, b in zip(scores, scores[1:]) if (a - b) > eps)
#     total_pairs = T - 1
#
#     strict_inc = inc_steps / total_pairs
#     nondec = (inc_steps + sum(1 for a, b in zip(scores, scores[1:]) if abs(b - a) <= eps)) / total_pairs
#
#     # Spearman
#     rho = _spearman_rank_corr(scores)
#     # First-to-last delta
#     delta = scores[-1] - scores[0]
#     # Violations
#     v = sum(1 for a, b in zip(scores, scores[1:]) if b < a)
#     # AST coverage
#     cov = sources.count("AST") / T
#
#     return {
#         "scores": scores, "sources": sources,
#         "adjacent_strict_increasing": strict_inc,
#         "adjacent_non_decreasing": nondec,
#         "spearman_r": rho,
#         "first_last_delta": delta,
#         "violations": dec_steps,
#         "ast_coverage": sources.count("AST") / T
#     }