#!/usr/bin/env python3
"""
Create a PSA score vs corpus size plot for each grammar/corpus type.

Conventions (kept consistent across all your figures):
- X-axis: corpus sizes [100, 200, 300, 400, 500]
- Y-axis: metric score (here, PSA score)
- Colours: balanced=blue, high_constraint=orange, loose=green
- Lines: thick and easy to read, with clear markers
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---- fixed inputs ------------------------------------------------------------

csv_path = Path("psa_evaluation_summary_20250916_052750.csv")
output_path = Path("psa_score_vs_size.png")

# Column names expected in the CSV
grammar_col = "grammar"
size_col = "size"
metric_col = "psa_score"  # change this if you plot a different metric later

# Sizes to display on the x-axis (others will be ignored)
x_sizes = [100, 200, 300, 400, 500]

# Fixed colour mapping (must never change)
colour_map = {
    "balanced": "#1f77b4", # blue
    "high_constraint": "#ff7f0e", # orange
    "loose": "#2ca02c", # green
}

# Order in which to draw the curves (also fixes legend order)
draw_order = ["balanced", "high_constraint", "loose"]


def main() -> None:
    # ---- load & tidy ---------------------------------------------------------
    df = pd.read_csv(csv_path)

    # Keep only the sizes we wish to show and sort for neatness
    plot_df = (
        df[df[size_col].isin(x_sizes)]
        .copy()
        .sort_values([grammar_col, size_col])
    )

    # ---- plot ---------------------------------------------------------------
    plt.figure(figsize=(10, 7))

    for grammar in draw_order:
        sub = plot_df[plot_df[grammar_col] == grammar]
        if sub.empty:
            # Gracefully skip if a series is missing from the data
            continue

        # Plot with thick lines and clearly visible markers
        plt.plot(
            sub[size_col],
            sub[metric_col],
            label=grammar,
            linewidth=3.0,
            marker="o",
            markersize=8,
            c=colour_map[grammar],
        )

    # Axis labelling and ticks
    plt.xlabel("Corpus size")
    plt.ylabel("PSA score")
    plt.title("PSA Score vs Corpus Size")
    plt.xticks(x_sizes)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
