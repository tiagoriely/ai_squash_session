# plot_complexity_line.py
# Usage:
#   python plot_complexity_line.py \
#     --input 05_hybrid_retrievers_detailed_results.csv \
#     --out plots/performance_by_complexity_line.png

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def natural_sort_complex_ids(ids):
    """Sort like complex_01_cg < complex_02_cg < complex_03_cg …"""
    def key(x):
        m = re.search(r"complex_(\d+)_cg", str(x))
        return int(m.group(1)) if m else 10**9
    return sorted(ids, key=key)


def get_top1_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (query_id, strategy_name) with the rank==1 score."""
    top1 = (
        df[df["rank"] == 1]
        .groupby(["query_id", "strategy_name"], as_index=False)["fusion_score"]
        .first()
        .rename(columns={"fusion_score": "top1_score"})
    )
    return top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="05_hybrid_retrievers_detailed_results.csv")
    ap.add_argument("--out", default="plots/performance_by_complexity_line.png")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Top-1 per query/strategy, keep only complex_*_cg
    top1 = get_top1_per_query(df)
    top1 = top1[top1["query_id"].str.contains(r"^complex_\d+_cg$", regex=True)]

    if top1.empty:
        raise RuntimeError("No query_ids matching ^complex_\\d+_cg$ were found.")

    # X-axis order
    x_order = natural_sort_complex_ids(top1["query_id"].unique().tolist())
    x_idx = np.arange(len(x_order))

    # Plot (one line per retriever)
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, grp in top1.groupby("strategy_name"):
        y_vals = []
        for qid in x_order:
            row = grp.loc[grp["query_id"] == qid, "top1_score"]
            y_vals.append(row.iloc[0] if not row.empty else np.nan)
        ax.plot(x_idx, y_vals, marker="o", linestyle="-", label=strat)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_order, rotation=30, ha="right")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Top-1 Score")
    ax.set_title("Performance by Complexity (Top-1 score)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Retriever", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
