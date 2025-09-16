#!/usr/bin/env python3
"""
Diversity plots from 'selfbleu_distinctn_all_results.csv'

Outputs
-------
1) 'diversity_metrics_vs_size.png'
   • Three side-by-side curves (Self BLEU, Distinct-n, Combined Diversity Score)
   • Legend placed below the plots (no overlap with titles)

2) 'diversity_metric_averages_by_corpus.png'
   • Three **separate bar subplots** (one per metric) on the same image
   • Each subplot uses a **tight y-scale** so small differences are visible
   • Bars coloured by corpus type; exact values annotated on the bars

Conventions
-----------
- X-axis (curves): corpus sizes [100, 200, 300, 400, 500]
- Y-axis: metric score
- Fixed colours (Tab10): balanced=#1f77b4, high_constraint=#ff7f0e, loose=#2ca02c
- Lines: thick with clear markers
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------- fixed inputs ------------------------------------------------------

csv_path  = Path("selfbleu_distinctn_all_results.csv")
out_curves = Path("lexical_diversity_metrics_vs_size.png")
out_bars   = Path("lexical_metric_averages_by_corpus.png")

# Column names
grammar_col     = "grammar_type"
size_col        = "size"
col_self_bleu   = "self_bleu"
col_distinct    = "avg_distinct_n"
col_combined    = "combined_diversity_score"

# Sizes shown on x-axis (others ignored)
x_sizes = [100, 200, 300, 400, 500]

# Canonical mapping (normalise CSV variants)
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Exact Tab10 colours
colour_map = {
    "balanced": "#1f77b4",        # tab:blue
    "high_constraint": "#ff7f0e", # tab:orange
    "loose": "#2ca02c",           # tab:green
}
draw_order = ["balanced", "high_constraint", "loose"]

# Small horizontal jitter so coincident points/lines remain visible
x_jitter = {"balanced": -3.0, "high_constraint": 0.0, "loose": 3.0}


def _tight_ylim(ax: plt.Axes, values: list[float]) -> None:
    """Set a tighter y-limit with sensible padding so small differences pop."""
    vals = np.asarray(values, dtype=float)
    ymin, ymax = float(np.min(vals)), float(np.max(vals))
    rng = ymax - ymin
    # Ensure a minimum span so bars aren't clipped; more precise if range is tiny
    min_span = max(0.01, 0.2 * ymin if ymin > 0 else 0.02)
    span = max(rng, min_span)
    pad = 0.15 * span
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _fmt_value(x: float) -> str:
    """Nicely format numbers; 3 decimals for values < 1, else 2 decimals."""
    return f"{x:.3f}" if abs(x) < 1 else f"{x:.2f}"


def main() -> None:
    # -------- load & tidy -----------------------------------------------------
    df = pd.read_csv(csv_path)

    # Normalise grammar labels
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )

    # Keep only the desired sizes
    df = df[df[size_col].isin(x_sizes)].copy()

    metrics = [col_self_bleu, col_distinct, col_combined]

    # Aggregate means per (grammar, size) in case of repeated runs
    agg = df.groupby(["grammar_canon", size_col], as_index=False)[metrics].mean()

    # -------- Figure 1: curves (legend below, not on titles) ------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles  = ["Self BLEU vs Corpus Size", "Distinct-n vs Corpus Size", "Combined Diversity Score vs Corpus Size"]
    ylabels = ["Self BLEU", "Distinct-n", "Combined Diversity Score"]

    for ax, title, col, ylabel in zip(axes, titles, metrics, ylabels):
        for corpus in draw_order:
            sub = agg[agg["grammar_canon"] == corpus]
            if sub.empty:
                continue
            x = sub[size_col].astype(float) + x_jitter[corpus]
            ax.plot(
                x,
                sub[col],
                label=corpus,
                linewidth=3.0,
                marker="o",
                markersize=8,
                c=colour_map[corpus],
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Corpus size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_sizes)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Single legend placed beneath the axes (so it never overlaps titles)
    handles = [plt.Line2D([0], [0], color=colour_map[c], lw=3) for c in draw_order]
    fig.legend(
        handles, draw_order,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.subplots_adjust(bottom=0.16)  # make space for the legend

    plt.tight_layout()
    plt.savefig(out_curves, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved curves to: {out_curves.resolve()}")

    # -------- Figure 2: three bar subplots (each with tight y-scale) ----------
    means = df.groupby("grammar_canon", as_index=False)[metrics].mean()

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    metric_cols    = [col_self_bleu, col_distinct, col_combined]
    metric_pretty  = ["Self BLEU", "Distinct-n", "Combined Diversity"]

    for ax, col, pretty in zip(axes2, metric_cols, metric_pretty):
        heights = [
            float(means.loc[means["grammar_canon"] == c, col].squeeze())
            if c in set(means["grammar_canon"].values) else 0.0
            for c in draw_order
        ]
        bars = ax.bar(
            np.arange(len(draw_order)),
            heights,
            color=[colour_map[c] for c in draw_order],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(np.arange(len(draw_order)))
        ax.set_xticklabels(draw_order)
        ax.set_ylabel("Average score")
        ax.set_title(f"{pretty} — Average by Corpus")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Tighten y-axis for better visual separation
        _tight_ylim(ax, heights)

        # Annotate exact values on bars
        for bar, val in zip(bars, heights):
            ax.annotate(
                _fmt_value(val),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_bars, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved bar subplots to: {out_bars.resolve()}")


if __name__ == "__main__":
    main()
