# evaluation/corpus_analysis/statistics/measure_structure.py

import argparse
from pathlib import Path
import math
import re
from typing import List, Dict, Any

from collections import Counter, defaultdict

# Import the shared utility and metric functions
from evaluation.corpus_analysis.utils import load_corpus
from evaluation.corpus_analysis.statistics.common_metrics import calculate_shannon_entropy


# --- Metric Helper Functions ---
def analyse_mode_interleaving(corpus):
    """
    Compute the Mode Interleaving Index over activity exercises, using the rendered session text.

    Purpose
    -------
    Quantifies how frequently a session alternates between 'drill' (minutes) and
    'conditioned game' (points) across the ordered list of activity exercises.
    Higher values indicate more interleaving; lower values indicate longer runs
    of the same mode (more regimented blocks).

    Definition
    ----------
    Let m_1, ..., m_N be the ordered modes for all *activity* exercises in a session,
    where m_i ∈ {drill, conditioned_game}. The per-session interleaving index is:
        I = (1 / (N - 1)) * Σ_{i=1..N-1} [ m_i ≠ m_{i+1} ]
    The corpus-level index is the average of per-session indices (weighted by N-1).

    Parameters
    ----------
    corpus : Iterable[dict]
        JSONL-like list of session records, each with "contents" (rendered text).

    Returns
    -------
    dict
        {
          "mean_interleaving": float in [0,1],
          "sessions_covered": int,
          "weighted_interleaving": float in [0,1],   # weighted by (N-1)
          "summary": {
              "num_sessions_with_<2_items>": int,
              "mean_run_length": float              # average run length of same-mode streaks
          }
        }
    """

    bullet = re.compile(r"^\s*[•\-\*]\s")
    pts_pat = re.compile(r"\bpt[s]?:", re.IGNORECASE)
    min_pat = re.compile(r"\bmin:", re.IGNORECASE)

    def extract_ordered_modes_from_contents(txt: str):
        modes = []
        for raw in (txt or "").splitlines():
            line = raw.strip()
            if not bullet.match(line):
                continue
            if "Rest:" in line:
                continue
            # Skip warm-ups explicitly
            if "Warmup:" in line or "Warm-up" in line:
                continue
            if pts_pat.search(line):
                modes.append("conditioned_game")
            elif min_pat.search(line):
                modes.append("drill")
        return modes

    total_weight, weighted_sum = 0, 0.0
    per_session = []
    run_lengths_all = []

    for session in corpus:
        modes = extract_ordered_modes_from_contents(session.get("contents", ""))
        if len(modes) < 2:
            per_session.append(0.0)
            continue
        # Interleaving
        diffs = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
        denom = len(modes) - 1
        idx = diffs / denom
        per_session.append(idx)
        total_weight += denom
        weighted_sum += diffs

        # Run-lengths for interpretability
        run = 1
        for a, b in zip(modes, modes[1:]):
            if a == b:
                run += 1
            else:
                run_lengths_all.append(run)
                run = 1
        run_lengths_all.append(run)

    mean_interleaving = sum(per_session) / max(1, len(per_session))
    weighted_interleaving = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    mean_run_length = (sum(run_lengths_all) / len(run_lengths_all)) if run_lengths_all else 0.0

    sessions_too_short = 0
    for session in corpus:
        if len(extract_ordered_modes_from_contents(session.get("contents", ""))) < 2:
            sessions_too_short += 1

    return {
        "mean_interleaving": mean_interleaving,
        "sessions_covered": len(per_session),
        "weighted_interleaving": weighted_interleaving,
        "summary": {
            "sessions_too_short": sessions_too_short,
            "mean_run_length": mean_run_length
        }
    }


def calculate_distributions(corpus: List[Dict]) -> tuple[Counter, Counter]:
    """Counts the occurrences of archetypes and structure IDs in a corpus."""
    archetype_counts = Counter()
    structure_counts = Counter()

    for session in corpus:
        meta = session.get("meta", {})
        if archetype := meta.get("archetype"):
            archetype_counts[archetype] += 1
        if structure_id := meta.get("structure_id"):
            structure_counts[structure_id] += 1

    return archetype_counts, structure_counts


def analyse_family_transitions(corpus):
    """
    Estimate macro-structural flow between exercise families using an order-aware Markov model.

    Purpose
    -------
    Measures how predictable the 'next family' is given the 'current family' across the
    ordered list of 'activity' exercises. Lower conditional entropy indicates more regimented,
    rule-like session flow; higher entropy indicates looser, more varied flow.

    Definitions
    -----------
    From ordered family IDs f_1, ..., f_M (warm-ups excluded), collect bigram counts C(s,a)
    for transitions s→a. Let:
        P(a|s) = C(s,a) / Σ_a C(s,a)     and     P(s) = Σ_a C(s,a) / Σ_{s,a} C(s,a)

    Conditional entropy (bits):
        H = Σ_s P(s) * H(P(·|s))    where  H(P) = -Σ_x P(x) log2 P(x)

    Normalised per-state entropy:
        H_norm = Σ_s P(s) * ( H(P(·|s)) / log2(outdegree(s)) )

    Determinism index:
        D = Σ_s P(s) * max_a P(a|s)

    Parameters
    ----------
    corpus : Iterable[dict]
        JSONL-like list of session records with meta.exercise_sequences.

    Returns
    -------
    dict
        {
          "top_transitions": List[Tuple[(s,a), prob]],
          "conditional_entropy_bits": float,
          "conditional_entropy_norm": float,
          "determinism_index": float
        }
    """
    transition_counts = defaultdict(lambda: Counter())

    # Build ordered family bigrams from exercise_sequences, excluding warm-ups
    for session in corpus:
        seq_entries = session.get("meta", {}).get("exercise_sequences", []) or []
        families = [
            ex.get("exercise_family_id")
            for ex in seq_entries
            if ex and ex.get("exercise_family_id") and ex.get("exercise_family_id") != "squash.family.warmup"
        ]
        for a, b in zip(families, families[1:]):
            transition_counts[a][b] += 1

    # Transition probabilities and per-state metrics
    transition_probabilities = defaultdict(dict)
    state_masses, state_entropies, state_entropies_norm, state_max_probs = {}, {}, {}, {}
    total_transitions = sum(sum(c.values()) for c in transition_counts.values()) or 0

    for s, next_counts in transition_counts.items():
        total_from_s = sum(next_counts.values())
        for a, c in next_counts.items():
            transition_probabilities[s][a] = c / total_from_s

    for s, next_probs in transition_probabilities.items():
        probs = list(next_probs.values())
        Hs = calculate_shannon_entropy(probs)
        outdeg = max(1, len(probs))
        Hmax = math.log2(outdeg)
        state_entropies[s] = Hs
        state_entropies_norm[s] = (Hs / Hmax) if Hmax > 0 else 0.0
        state_max_probs[s] = max(probs) if probs else 0.0

    for s, next_counts in transition_counts.items():
        state_masses[s] = (sum(next_counts.values()) / total_transitions) if total_transitions > 0 else 0.0

    conditional_entropy = sum(state_masses[s] * state_entropies[s] for s in state_entropies)
    conditional_entropy_norm = sum(state_masses[s] * state_entropies_norm[s] for s in state_entropies_norm)
    determinism_index = sum(state_masses[s] * state_max_probs[s] for s in state_max_probs)

    # Top transitions for interpretability
    all_pairs = []
    for s, d in transition_probabilities.items():
        for a, p in d.items():
            all_pairs.append(((s, a), p))
    top_transitions = sorted(all_pairs, key=lambda x: x[1], reverse=True)[:10]

    return {
        "top_transitions": top_transitions,
        "conditional_entropy_bits": conditional_entropy,
        "conditional_entropy_norm": conditional_entropy_norm,
        "determinism_index": determinism_index
    }

def analyse_ast_richness(corpus):
    """
    Measure within-exercise grammar richness from sequence ASTs (activities only).

    Purpose
    -------
    Captures the 'micro-structural' shape permitted by the DSL: branching (Choice),
    looping (Repeat), optional paths (Optional), and explicit control (Restart).
    Reported as rates per visited AST node to be comparable across corpora.

    Definitions
    -----------
    For each activity exercise with a parsed AST, traverse nodes (including children):
      counts[k] = number of nodes of type k,  k ∈ {Choice, Repeat, Optional, Restart}
      total     = total number of visited nodes
    Then rate_k = counts[k] / total

    Parameters
    ----------
    corpus : Iterable[dict]
        JSONL-like list of session records with meta.exercise_sequences[*].sequence_ast.

    Returns
    -------
    dict
        {
          "ast_rate_choice": float in [0,1],
          "ast_rate_repeat": float in [0,1],
          "ast_rate_optional": float in [0,1],
          "ast_rate_restart": float in [0,1]
        }
    """
    wanted = {"Choice", "Repeat", "Optional", "Restart"}
    counts = Counter({k: 0 for k in wanted})
    total_nodes = 0

    for session in corpus:
        for ex in session.get("meta", {}).get("exercise_sequences", []) or []:
            fam = ex.get("exercise_family_id")
            if not fam or fam == "squash.family.warmup":
                continue
            ast = ex.get("sequence_ast")
            if not ast:
                continue
            stack = list(ast)
            while stack:
                node = stack.pop()
                ntype = node.get("type")
                if ntype in wanted:
                    counts[ntype] += 1
                total_nodes += 1
                if ntype == "Choice":
                    stack.extend(node.get("options", []))
                elif ntype in ("Repeat", "Optional"):
                    body = node.get("body")
                    if isinstance(body, dict):
                        stack.append(body)

    denom = max(1, total_nodes)
    return {
        "ast_rate_choice": counts["Choice"] / denom,
        "ast_rate_repeat": counts["Repeat"] / denom,
        "ast_rate_optional": counts["Optional"] / denom,
        "ast_rate_restart": counts["Restart"] / denom,
    }


def analyse_template_adherence(corpus: List[Dict]) -> Dict[str, Any]:
    """
    Calculates the adherence of sessions to their declared structural templates.

    - 'Drill Only' sessions should only contain 'drill' modes.
    - 'Conditioned Games Only' sessions should only contain 'conditioned_game' modes.
    - 'Dynamic Block' sessions should contain at least one of each.
    """
    adherence_counts = {
        "drill_only": {"total": 0, "valid": 0},
        "cg_only": {"total": 0, "valid": 0},
        "dynamic_block": {"total": 0, "valid": 0},
    }

    # Re-use the mode extraction logic from analyse_mode_interleaving
    bullet = re.compile(r"^\s*[•\-\*]\s")
    pts_pat = re.compile(r"\bpt[s]?:", re.IGNORECASE)
    min_pat = re.compile(r"\bmin:", re.IGNORECASE)

    for session in corpus:
        meta = session.get("meta", {})
        archetype = meta.get("archetype", "").lower()

        # Extract modes
        modes = []
        for raw in (session.get("contents", "") or "").splitlines():
            line = raw.strip()
            if not bullet.match(line) or "Warmup:" in line or "Warm-up" in line:
                continue
            if pts_pat.search(line):
                modes.append("conditioned_game")
            elif min_pat.search(line):
                modes.append("drill")

        if not modes:
            continue

        # Check adherence based on archetype
        if "drill only" in archetype:
            adherence_counts["drill_only"]["total"] += 1
            if all(m == "drill" for m in modes):
                adherence_counts["drill_only"]["valid"] += 1

        elif "conditioned games only" in archetype:
            adherence_counts["cg_only"]["total"] += 1
            if all(m == "conditioned_game" for m in modes):
                adherence_counts["cg_only"]["valid"] += 1

        elif "dynamic block" in archetype:
            adherence_counts["dynamic_block"]["total"] += 1
            mode_set = set(modes)
            if "drill" in mode_set and "conditioned_game" in mode_set:
                adherence_counts["dynamic_block"]["valid"] += 1

    # Calculate overall adherence rate
    total_relevant_sessions = sum(v["total"] for v in adherence_counts.values())
    total_valid_sessions = sum(v["valid"] for v in adherence_counts.values())

    overall_rate = (total_valid_sessions / total_relevant_sessions) if total_relevant_sessions > 0 else 0.0

    return {
        "details": adherence_counts,
        "overall_adherence_rate": overall_rate
    }

# --- Main Analysis Orchestrator ---

def analyse_structure_metrics(corpus):
    """
    Aggregate macro- and micro-structure metrics for a corpus.

    Returns a dictionary combining:
      - composition summaries (archetype/structure distributions),
      - macro flow metrics (family-flow conditional entropy, determinism,
        and mode interleaving),
      - micro grammar metrics (AST richness rates).
    """
    archetype_counts, structure_counts = calculate_distributions(corpus)


    transitions = analyse_family_transitions(corpus)
    interleaving = analyse_mode_interleaving(corpus)
    ast_stats = analyse_ast_richness(corpus)
    template_adherence = analyse_template_adherence(corpus)


    return {
        "archetype_distribution": dict(archetype_counts),
        "structure_distribution": dict(structure_counts),
        "transition_top10": transitions["top_transitions"],
        "transition_conditional_entropy_bits": transitions["conditional_entropy_bits"],
        "transition_conditional_entropy_norm": transitions["conditional_entropy_norm"],
        "transition_determinism_index": transitions["determinism_index"],
        "mode_interleaving_mean": interleaving["mean_interleaving"],
        "mode_interleaving_weighted": interleaving["weighted_interleaving"],
        "mode_run_length_mean": interleaving["summary"]["mean_run_length"],
        "template_adherence_rate": template_adherence["overall_adherence_rate"],
        **ast_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the structural properties of a generated corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("--json-indent", type=int, default=2, help="Pretty-print JSON with this indent.")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    results = analyse_structure_metrics(corpus)

    # Print to stdout so this can be piped/saved
    import json
    print(json.dumps(results, indent=args.json_indent, ensure_ascii=False))