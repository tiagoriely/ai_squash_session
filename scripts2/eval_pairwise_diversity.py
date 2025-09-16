# scripts2/eval_pairwise_diversity.py
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ---------- repo paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ---------- evaluator ----------
from evaluation.evaluators.llm_judge import LlmEvaluator

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


def parse_case_id(case_id: str) -> tuple[str, int | None]:
    """Parses a case_id like 'loose_100_query_k10' into ('loose', 100)."""
    parts = case_id.split("_")
    grammar = parts[0]
    size = None
    for p in parts[1:]:
        if p.isdigit():
            size = int(p)
            break
    return grammar, size


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

    parsed_ids = df["case_id"].apply(parse_case_id)
    df["grammar"] = [g for g, s in parsed_ids]
    df["size"] = [s for g, s in parsed_ids]

    if "query_text" not in df.columns:
        df["query_text"] = "<unknown-query>"

    df["row_id"] = [f"{g}-{s if s else 'na'}-{i}" for i, (g, s) in enumerate(parsed_ids)]

    judge = LlmEvaluator(
        model_name="gpt-4-turbo-preview",
        prompts_dir=PROJECT_ROOT / "evaluation" / "prompts"
    )

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
    grouped_q = df.groupby("query_text")
    within_results = []
    cross_results = []

    print("\nProcessing pairs... (this may take a while)")
    for query, gdf in grouped_q:
        ids_by_group = {gs: grp["row_id"].tolist() for gs, grp in gdf.groupby(["grammar", "size"])}

        within_pairs_to_process = []
        for (gram, size), ids in ids_by_group.items():
            if len(ids) < 2:
                continue
            unique_ids_in_pairs = set(
                itertools.chain.from_iterable(_sample_pairs(ids, within_pairs_per_query_per_grammar, rng)))
            for an_id in unique_ids_in_pairs:
                if an_id not in sig_cache:
                    plan_text = df.loc[df["row_id"] == an_id, "generated_plan"].iloc[0]
                    sig_cache[an_id] = extract_signature(plan_text)

            for a, b in _sample_pairs(ids, within_pairs_per_query_per_grammar, rng):
                within_pairs_to_process.append(((gram, size), a, b))

        for (gram, size), a, b in tqdm(within_pairs_to_process,
                                       desc=f"Judging WITHIN pairs for query: {query[:30]}..."):
            sig_a = sig_cache[a]
            sig_b = sig_cache[b]
            dec = judge_pair(sig_a, sig_b)

            within_results.append({"query": query, "grammar": gram, "size": int(size), "a": a, "b": b, **dec})

        ids_by_grammar = {g: grp["row_id"].tolist() for g, grp in gdf.groupby("grammar")}
        cross_pairs = _sample_cross_pairs(ids_by_grammar, cross_pairs_per_query, rng)
        unique_ids_in_cross_pairs = set(itertools.chain.from_iterable(cross_pairs))
        for an_id in unique_ids_in_cross_pairs:
            if an_id not in sig_cache:
                plan_text = df.loc[df["row_id"] == an_id, "generated_plan"].iloc[0]
                sig_cache[an_id] = extract_signature(plan_text)

        for a, b in tqdm(cross_pairs, desc=f"Judging CROSS pairs for query: {query[:30]}..."):
            ra = df.loc[df["row_id"] == a].iloc[0]
            rb = df.loc[df["row_id"] == b].iloc[0]
            sig_a = sig_cache[a]
            sig_b = sig_cache[b]
            dec = judge_pair(sig_a, sig_b)
            cross_results.append(
                {"query": query, "grammar_a": ra["grammar"], "grammar_b": rb["grammar"], "a": a, "b": b, **dec})

    print("\n--- Summary generation ---")

    outdir = PROJECT_ROOT / "experiments" / "pairwise_diversity"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "within_pairs.json").write_text(json.dumps(within_results, indent=2), encoding="utf-8")
    (outdir / "cross_pairs.json").write_text(json.dumps(cross_results, indent=2), encoding="utf-8")
    print(f"\nSaved raw decisions to {outdir}/")

    if within_results:
        df_within = pd.DataFrame(within_results)
        grouping_keys = ['grammar', 'size']

        mean_scores = df_within.groupby(grouping_keys)['overall_distinctness'].mean().reset_index()
        mean_scores = mean_scores.rename(columns={'overall_distinctness': 'mean_diversity_score'})

        score_proportions = df_within.groupby(grouping_keys)['overall_distinctness'] \
                                .value_counts(normalize=True) \
                                .unstack(fill_value=0) \
                                .reindex(columns=[0, 1, 2], fill_value=0) * 100

        score_proportions.columns = ['percent_score_0', 'percent_score_1', 'percent_score_2']

        summary_df = pd.merge(mean_scores, score_proportions, on=grouping_keys)
        summary_df = summary_df.sort_values(by=['grammar', 'size'])

        csv_output_path = outdir / "pairwise_summary.csv"
        summary_df.to_csv(csv_output_path, index=False)
        print(f"✅ Summary CSV (with size) saved to: {csv_output_path}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Pairwise LLM-as-judge diversity evaluation.")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSON file with generated plans.")
    parser.add_argument("--within", type=int, default=WITHIN_PAIRS_PER_QUERY_PER_GRAMMAR,
                        help="Within-corpus pairs per (query, grammar).")
    parser.add_argument("--cross", type=int, default=CROSS_PAIRS_PER_QUERY, help="Cross-corpus pairs per query.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()
    run_pairwise(args.input, args.within, args.cross, args.seed)