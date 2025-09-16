"""
Create an Average Diversity Score vs corpus size plot for each grammar/corpus type.

Conventions (kept consistent across all your figures):
- X-axis: corpus sizes [100, 200, 300, 400, 500]
- Y-axis: metric score (here, Average Diversity Score)
- Colours: balanced=#1f77b4, high_constraint=#ff7f0e, loose=#2ca02c (Tab10)
- Lines: thick and easy to read, with clear markers
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---- fixed inputs ------------------------------------------------------------

csv_path = Path("pairwise_summary_v1.csv")
output_path = Path("avg_diversity_vs_size.png")

# Column names expected in the CSV
grammar_col = "grammar"
size_col = "size"
metric_col = "mean_diversity_score"  # maps to “Average Diversity Score”

# Sizes to display on the x-axis (others will be ignored)
x_sizes = [100, 200, 300, 400, 500]

# Canonical label mapping (normalises any variants from the CSV)
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",          # <- map 'high' to 'high_constraint'
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Fixed colour mapping (must never change) — exact Tab10 hexes
colour_map = {
    "balanced": "#1f77b4",
    "high_constraint": "#ff7f0e",
    "loose": "#2ca02c",
}

# Order in which to draw the curves (also fixes legend order)
draw_order = ["balanced", "high_constraint", "loose"]


def main() -> None:
    # ---- load & tidy ---------------------------------------------------------
    df = pd.read_csv(csv_path)

    # Normalise the grammar labels to canonical ones
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )

    # Keep only the sizes we wish to show and aggregate if duplicates exist
    plot_df = (
        df[df[size_col].isin(x_sizes)]
        .groupby(["grammar_canon", size_col], as_index=False)[metric_col]
        .mean()
        .sort_values(["grammar_canon", size_col])
    )

    # ---- plot ----------------------------------------------------------------
    plt.figure(figsize=(10, 7))

    for label in draw_order:
        sub = plot_df[plot_df["grammar_canon"] == label]
        if sub.empty:
            continue

        plt.plot(
            sub[size_col],
            sub[metric_col],
            label=label,                  # legend shows 'high_constraint'
            linewidth=3.0,
            marker="o",
            markersize=8,
            c=colour_map[label],          # colour tied to canonical label
        )

    # Axis labelling and ticks
    plt.xlabel("Corpus size")
    plt.ylabel("Average Diversity Score")
    plt.title("Average Diversity (Pairwise LLM Judge) Score vs Corpus Size")
    plt.xticks(x_sizes)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
