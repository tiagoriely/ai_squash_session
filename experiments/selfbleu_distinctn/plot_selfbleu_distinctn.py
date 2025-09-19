#!/usr/bin/env python3
"""
Make SIX separate figures (default Matplotlib styling) from
'selfbleu_distinctn_all_results.csv':

Curves (one file each):
  1) 'self_bleu_vs_size.png'            — Self BLEU vs corpus size
  2) 'distinct_n_vs_size.png'           — Distinct-n vs corpus size
  3) 'combined_diversity_vs_size.png'   — Combined Diversity Score vs corpus size

Bars (one file each; averaged across sizes):
  4) 'self_bleu_avg_by_corpus.png'      — with a tighter y-scale for precision
  5) 'distinct_n_avg_by_corpus.png'
  6) 'combined_diversity_avg_by_corpus.png'

Conventions:
- X-axis (curves): corpus sizes [50, 100, 200, 300, 400, 500]
- Colours (fixed): balanced=#1f77b4, high_constraint=#ff7f0e, loose=#2ca02c
- Grammar normalisation: 'high' → 'high_constraint'
- Default line thickness and default font sizes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- fixed inputs/outputs ---------------------------------------------

csv_path = Path("selfbleu_distinctn_all_results.csv")

out_self_bleu_curve = Path("self_bleu_vs_size.png")
out_distinct_curve  = Path("distinct_n_vs_size.png")
out_combined_curve  = Path("combined_diversity_vs_size.png")

out_self_bleu_bars = Path("self_bleu_avg_by_corpus.png")
out_distinct_bars  = Path("distinct_n_avg_by_corpus.png")
out_combined_bars  = Path("combined_diversity_avg_by_corpus.png")

# Column names in the CSV
grammar_col   = "grammar_type"
size_col      = "size"
col_self_bleu = "self_bleu"
col_distinct  = "avg_distinct_n"
col_combined  = "combined_diversity_score"

# X-axis sizes to show
x_sizes = [50, 100, 200, 300, 400, 500]

# Canonical mapping for grammar values
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Exact Tab10 colours (do not change)
colour_map = {
    "balanced": "#1f77b4",
    "high_constraint": "#ff7f0e",
    "loose": "#2ca02c",
}
draw_order = ["balanced", "high_constraint", "loose"]

# Small horizontal jitter to avoid perfectly overlapping points when curves coincide
x_jitter = {"balanced": -3.0, "high_constraint": 0.0, "loose": 3.0}


def plot_curve(df: pd.DataFrame, metric_col: str, ylabel: str, title: str, outfile: Path) -> None:
    """Plot a single metric vs size (one figure). Default Matplotlib thickness/text."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for corpus in draw_order:
        sub = df[df["grammar_canon"] == corpus]
        if sub.empty:
            continue
        x = sub[size_col].astype(float) + x_jitter[corpus]
        ax.plot(
            x,
            sub[metric_col],
            marker="o",
            label=corpus,
            c=colour_map[corpus],
        )

    ax.set_title(title)
    ax.set_xlabel("Corpus size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_sizes)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars, values):
    """Write numeric values above bars."""
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_bars(means: pd.DataFrame, metric_col: str, pretty: str, outfile: Path,
              tighten_scale: bool = False) -> None:
    """Plot a single bar figure (average per corpus). Optionally tighten y-scale."""
    fig, ax = plt.subplots(figsize=(7, 5))

    heights = [
        float(means.loc[means["grammar_canon"] == c, metric_col].squeeze())
        if c in set(means["grammar_canon"].values) else 0.0
        for c in draw_order
    ]

    x = np.arange(len(draw_order))
    bars = ax.bar(
        x,
        heights,
        color=[colour_map[c] for c in draw_order],
        edgecolor="black",
        linewidth=0.6,
    )

    # Optional: tighter y-scale for better precision (used for Self BLEU)
    if tighten_scale:
        ymin, ymax = min(heights), max(heights)
        pad = 0.01                        # small headroom
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(draw_order)
    ax.set_ylabel("Average score")
    ax.set_title(f"{pretty} — Average by Corpus")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    _annotate_bars(ax, bars, heights)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Load and tidy
    df = pd.read_csv(csv_path)
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )
    df = df[df[size_col].isin(x_sizes)].copy()

    # Aggregate means per (grammar, size) for the curves (in case of repeats)
    metrics = [col_self_bleu, col_distinct, col_combined]
    agg = df.groupby(["grammar_canon", size_col], as_index=False)[metrics].mean()

    # --- Curves (three separate files)
    plot_curve(agg, col_self_bleu, "Self BLEU", "Self BLEU vs Corpus Size", out_self_bleu_curve)
    plot_curve(agg, col_distinct, "Distinct-n", "Distinct-n vs Corpus Size", out_distinct_curve)
    plot_curve(agg, col_combined, "Combined Diversity Score", "Combined Diversity Score vs Corpus Size", out_combined_curve)

    # --- Bar plots (averaged across sizes), three separate files
    means = df.groupby("grammar_canon", as_index=False)[metrics].mean()

    # Self BLEU bar plot: tighten y-scale so ~0.03 changes are clearly visible
    plot_bars(means, col_self_bleu, "Self BLEU", out_self_bleu_bars, tighten_scale=True)

    # Distinct-n and Combined: keep default scaling (distinct-n already fine)
    plot_bars(means, col_distinct, "Distinct-n", out_distinct_bars, tighten_scale=False)
    plot_bars(means, col_combined, "Combined Diversity", out_combined_bars, tighten_scale=False)

    print("Saved figures:")
    for p in [
        out_self_bleu_curve, out_distinct_curve, out_combined_curve,
        out_self_bleu_bars, out_distinct_bars, out_combined_bars,
    ]:
        print(" -", p.resolve())


if __name__ == "__main__":
    main()
