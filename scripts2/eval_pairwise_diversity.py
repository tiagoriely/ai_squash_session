# scripts2/eval_pairwise_diversity.py
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------- repo paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ---------- evaluator ----------
from evaluation.evaluators.llm_judge import LlmEvaluator  # uses OpenAI chat + json mode

# ---------- constants (can be overridden via CLI) ----------
WITHIN_PAIRS_PER_QUERY_PER_GRAMMAR = 50
CROSS_PAIRS_PER_QUERY = 50
RANDOM_SEED = 7


def _sample_pairs(ids, n_pairs, rng):
    allp = list(itertools.combinations(ids, 2))
    if len(allp) <= n_pairs:
        return allp
    return rng.sample(allp, n_pairs)


def _sample_cross_pairs(ids_by_grammar, n_pairs, rng):
    """Make cross-grammar pairs for the same query."""
    out = []
    grams = list(ids_by_grammar.keys())
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            A, B = grams[i], grams[j]
            a_ids, b_ids = ids_by_grammar[A], ids_by_grammar[B]
            if not a_ids or not b_ids:
                continue
            cross = list(itertools.product(a_ids, b_ids))
            out.extend(rng.sample(cross, min(n_pairs, len(cross))))
    return out


def run_pairwise(input_file: Path,
                 within_pairs_per_query_per_grammar: int = WITHIN_PAIRS_PER_QUERY_PER_GRAMMAR,
                 cross_pairs_per_query: int = CROSS_PAIRS_PER_QUERY,
                 random_seed: int = RANDOM_SEED):
    print(f"Loading generated sessions from: {input_file}")

    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    df = pd.DataFrame(data)

    if "generated_plan" not in df.columns:
        raise ValueError("Input JSON must contain 'generated_plan' for each item.")
    if "case_id" not in df.columns:
        raise ValueError("Input JSON must contain 'case_id' for each item.")

    # grammar from case_id (e.g., 'loose_500_complex_01_cg_k10' -> 'loose')
    def grammar_from_case(case_id: str) -> str:
        return case_id.split("_")[0]

    df["grammar"] = df["case_id"].apply(grammar_from_case)
    # Use query_text if present; else group everything under one bucket
    if "query_text" not in df.columns:
        df["query_text"] = "<unknown-query>"

    # Stable row id for caching
    df["row_id"] = [f"{g}-{i}" for i, g in enumerate(df["grammar"])]

    # ----- init judge (loads OPENAI_API_KEY via dotenv in its constructor too) -----
    judge = LlmEvaluator(
        model_name="gpt-4-turbo-preview",
        prompts_dir=PROJECT_ROOT / "evaluation" / "prompts"
    )

    # ----- cache signatures to avoid repeated extraction -----
    sig_cache: dict[str, dict] = {}

    def extract_signature(plan_text: str) -> dict:
        res = judge.extract_signature(plan_text)
        if isinstance(res, dict) and "error" in res:
            raise RuntimeError(f"LLM extract_signature error: {res['error']}")
        return res

    def judge_pair(sig_a: dict, sig_b: dict) -> dict:
        res = judge.judge_pairwise(
            sig_json_A=json.dumps(sig_a, ensure_ascii=False),
            sig_json_B=json.dumps(sig_b, ensure_ascii=False),
        )
        if isinstance(res, dict) and "error" in res:
            raise RuntimeError(f"LLM judge_pairwise error: {res['error']}")
        return res

    rng = random.Random(random_seed)

    # ----- build pairs per query -----
    grouped_q = df.groupby("query_text")

    within_results = []   # pairs within the same grammar for a query
    cross_results = []    # pairs across grammars for a query

    for query, gdf in grouped_q:
        ids_by_grammar = {g: grp["row_id"].tolist() for g, grp in gdf.groupby("grammar")}

        # WITHIN
        for gram, ids in ids_by_grammar.items():
            if len(ids) < 2:
                continue
            for a, b in _sample_pairs(ids, within_pairs_per_query_per_grammar, rng):
                ra = df.loc[df["row_id"] == a].iloc[0]
                rb = df.loc[df["row_id"] == b].iloc[0]

                # signatures (cached)
                if a in sig_cache:
                    sig_a = sig_cache[a]
                else:
                    sig_a = extract_signature(ra["generated_plan"])
                    sig_cache[a] = sig_a

                if b in sig_cache:
                    sig_b = sig_cache[b]
                else:
                    sig_b = extract_signature(rb["generated_plan"])
                    sig_cache[b] = sig_b

                dec = judge_pair(sig_a, sig_b)
                within_results.append({
                    "query": query,
                    "grammar": gram,
                    "a": a,
                    "b": b,
                    **dec
                })

        # CROSS
        cross_pairs = _sample_cross_pairs(ids_by_grammar, cross_pairs_per_query, rng)
        for a, b in cross_pairs:
            ra = df.loc[df["row_id"] == a].iloc[0]
            rb = df.loc[df["row_id"] == b].iloc[0]

            sig_a = sig_cache.get(a)
            if sig_a is None:
                sig_a = extract_signature(ra["generated_plan"])
                sig_cache[a] = sig_a

            sig_b = sig_cache.get(b)
            if sig_b is None:
                sig_b = extract_signature(rb["generated_plan"])
                sig_cache[b] = sig_b

            dec = judge_pair(sig_a, sig_b)
            cross_results.append({
                "query": query,
                "grammar_a": ra["grammar"],
                "grammar_b": rb["grammar"],
                "a": a,
                "b": b,
                **dec
            })

    # ----- aggregate summaries -----
    def summarize(records, by_keys):
        if not records:
            return {}
        d = pd.DataFrame(records)
        facet_cols = ["diff_exercises", "diff_rules", "diff_motifs", "diff_blocks", "diff_side_focus"]
        out = {}
        for key, grp in d.groupby(by_keys):
            stats = {}
            for c in facet_cols:
                if c in grp:
                    stats[f"mean_{c}"] = grp[c].astype(float).mean()
            if "overall_distinctness" in grp:
                stats["frac_overall_2"] = (grp["overall_distinctness"] == 2).mean()
            stats["n_pairs"] = len(grp)
            out[key if isinstance(key, tuple) else (key,)] = stats
        return out

    within_summary = summarize(within_results, ["grammar"])
    cross_summary = summarize(cross_results, ["grammar_a", "grammar_b"])

    print("\n=== WITHIN-CORPUS (pooled across queries) ===")
    if within_summary:
        for (g,), s in within_summary.items():
            print(f"[{g}] n={s['n_pairs']}"
                  f"  overall_2={s.get('frac_overall_2', 0):.3f}"
                  f" | ex:{s.get('mean_diff_exercises', 0):.2f}"
                  f" rules:{s.get('mean_diff_rules', 0):.2f}"
                  f" motifs:{s.get('mean_diff_motifs', 0):.2f}"
                  f" blocks:{s.get('mean_diff_blocks', 0):.2f}"
                  f" side:{s.get('mean_diff_side_focus', 0):.2f}")
    else:
        print("No within-corpus pairs (need >=2 plans per (query, grammar)).")

    print("\n=== CROSS-CORPUS (same query, A vs B) ===")
    if cross_summary:
        for (ga, gb), s in cross_summary.items():
            print(f"[{ga} vs {gb}] n={s['n_pairs']}"
                  f"  overall_2={s.get('frac_overall_2', 0):.3f}"
                  f" | ex:{s.get('mean_diff_exercises', 0):.2f}"
                  f" rules:{s.get('mean_diff_rules', 0):.2f}"
                  f" motifs:{s.get('mean_diff_motifs', 0):.2f}"
                  f" blocks:{s.get('mean_diff_blocks', 0):.2f}"
                  f" side:{s.get('mean_diff_side_focus', 0):.2f}")
    else:
        print("No cross-corpus pairs (need at least two grammars for any query).")

    # ----- save raw judgments -----
    outdir = PROJECT_ROOT / "experiments" / "pairwise_diversity"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "within_pairs.json").write_text(json.dumps(within_results, indent=2), encoding="utf-8")
    (outdir / "cross_pairs.json").write_text(json.dumps(cross_results, indent=2), encoding="utf-8")
    print(f"\nSaved raw decisions to {outdir}/")


if __name__ == "__main__":
    load_dotenv()  # ensure OPENAI_API_KEY is loaded

    parser = argparse.ArgumentParser(description="Pairwise LLM-as-judge diversity evaluation.")
    parser.add_argument(
        "--input",
        type=Path,
        required=False,
        default=PROJECT_ROOT / "experiments" / "evaluation_sessions_set_k10_size500_20250907_180221.json",
        help="Path to JSON file with generated plans.",
    )
    parser.add_argument("--within", type=int, default=WITHIN_PAIRS_PER_QUERY_PER_GRAMMAR,
                        help="Within-corpus pairs per (query, grammar).")
    parser.add_argument("--cross", type=int, default=CROSS_PAIRS_PER_QUERY,
                        help="Cross-corpus pairs per query.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()

    run_pairwise(args.input, args.within, args.cross, args.seed)
