# export_hybrid_metrics.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def standardise_query_type(s: str) -> str:
    """Map raw query types into a compact set of groups."""
    if pd.isna(s):
        return "Other"
    s = str(s)
    if "Complexity" in s:
        return "High-Relevance (Complex)"
    if "Single Shotside" in s:
        return "High-Relevance (Shotside)"
    if "Vague" in s:
        return "Vague But Relevant"
    if "Other Sport" in s:
        return "Out-of-Scope (Other Sport)"
    return "Other"


def extract_complexity_level(s: str):
    """Return an integer complexity level if present, else None."""
    if pd.isna(s):
        return None
    m = re.search(r"[Cc]omplexity\s*(\d+)", str(s))
    return int(m.group(1)) if m else None


def compute_top1_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Top-1 Delta = score(rank 1) − score(rank 2) per query/strategy."""
    top2 = df[df["rank"].isin([1, 2])].copy()

    # Get rank1 and rank2 side-by-side
    piv = top2.pivot_table(
        index=["query_id", "strategy_name"], columns="rank", values="fusion_score", aggfunc="first"
    ).rename(columns={1: "rank1_score", 2: "rank2_score"}).reset_index()

    # If rank-2 missing, treat delta vs 0 (or NaN). Using 0 is safer/comparable.
    piv["rank2_score"] = piv["rank2_score"].fillna(0.0)
    piv["top1_delta"] = piv["rank1_score"] - piv["rank2_score"]
    return piv


def main():
    ap = argparse.ArgumentParser(description="Export hybrid retriever metrics to CSV.")
    ap.add_argument("--input", default="05_hybrid_retrievers_detailed_results.csv",
                    help="Path to detailed results CSV.")
    ap.add_argument("--outdir", default="outputs", help="Directory to write metric CSVs.")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")

    df = pd.read_csv(inp)

    # Keep a per-query map of types
    qtype_map = df[["query_id", "query_type"]].drop_duplicates().set_index("query_id")["query_type"]

    # ---------- 1) Per-query MAX score (within the top-K ranks you logged) ----------
    per_query_max = (
        df.groupby(["query_id", "strategy_name"], as_index=False)["fusion_score"]
        .max()
        .rename(columns={"fusion_score": "max_fusion_score"})
    )
    per_query_max["query_type"] = per_query_max["query_id"].map(qtype_map)
    per_query_max["query_type_grouped"] = per_query_max["query_type"].map(standardise_query_type)
    per_query_max["complexity_level"] = per_query_max["query_type"].map(extract_complexity_level)

    # ---------- CSV A: Average Max Score by Query Type ----------
    avg_max_by_type = (
        per_query_max.groupby(["strategy_name", "query_type_grouped"])["max_fusion_score"]
        .mean()
        .reset_index()
        .sort_values(["strategy_name", "query_type_grouped"])
    )
    avg_max_by_type.to_csv(outdir / "avg_max_score_by_query_type.csv", index=False)

    # ---------- CSV B: Performance by Graduated Complexity ----------
    perf_complex = (
        per_query_max[per_query_max["complexity_level"].notna()]
        .groupby(["strategy_name", "complexity_level"])["max_fusion_score"]
        .mean()
        .reset_index()
        .sort_values(["strategy_name", "complexity_level"])
    )
    perf_complex.to_csv(outdir / "performance_by_complexity.csv", index=False)

    # ---------- CSV C: Top-1 Delta (rank1 − rank2) ----------
    top1_delta = compute_top1_delta(df)
    top1_delta["query_type"] = top1_delta["query_id"].map(qtype_map)
    top1_delta["query_type_grouped"] = top1_delta["query_type"].map(standardise_query_type)
    # Full per-query output
    top1_delta[[
        "query_id", "strategy_name", "query_type", "query_type_grouped",
        "rank1_score", "rank2_score", "top1_delta"
    ]].to_csv(outdir / "top1_delta_per_query.csv", index=False)
    # Summary (mean) by group
    top1_summary = (
        top1_delta.groupby(["strategy_name", "query_type_grouped"])["top1_delta"]
        .mean()
        .reset_index()
        .sort_values(["strategy_name", "query_type_grouped"])
    )
    top1_summary.to_csv(outdir / "top1_delta_summary.csv", index=False)

    # ---------- CSV D: Distribution of Max Scores ----------
    # Global bins (stable across strategies)
    all_max = per_query_max["max_fusion_score"]
    # If scores are within [0,1], this gives neat bins; otherwise uses min/max from data.
    lo, hi = float(all_max.min()), float(all_max.max())
    if hi <= 1.0 and lo >= 0.0:
        bins = np.linspace(0.0, 1.0, 11)
    else:
        bins = np.linspace(lo, hi, 11)

    dist_rows = []
    for strat, grp in per_query_max.groupby("strategy_name"):
        counts, edges = np.histogram(grp["max_fusion_score"].values, bins=bins)
        for i in range(len(counts)):
            dist_rows.append({
                "strategy_name": strat,
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "count": int(counts[i])
            })
    pd.DataFrame(dist_rows).to_csv(outdir / "max_score_distribution.csv", index=False)

    # Optional: distribution by query type group (more granular)
    dist_rows_qt = []
    for (strat, qt), grp in per_query_max.groupby(["strategy_name", "query_type_grouped"]):
        counts, edges = np.histogram(grp["max_fusion_score"].values, bins=bins)
        for i in range(len(counts)):
            dist_rows_qt.append({
                "strategy_name": strat,
                "query_type_grouped": qt,
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "count": int(counts[i])
            })
    pd.DataFrame(dist_rows_qt).to_csv(outdir / "max_score_distribution_by_query_type.csv", index=False)

    print(f"CSV exports written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
