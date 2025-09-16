"""
Score distribution overview in one PNG:

Row 1: line plots of % Score==0/1/2 vs size (impact of size)
Row 2: bar plots of % Score==0/1/2 by corpus (averaged across sizes)

Colours: balanced=#1f77b4, high_constraint=#ff7f0e, loose=#2ca02c
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---- fixed inputs ------------------------------------------------------------
csv_path = Path("pairwise_summary_v1.csv")
output_path = Path("score_distribution_overview.png")

grammar_col = "grammar"
size_col = "size"
score_cols = ["percent_score_0", "percent_score_1", "percent_score_2"]
x_sizes = [100, 200, 300, 400, 500]

# Canonical label mapping
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Exact Tab10 colours
colour_map = {
    "balanced": "#1f77b4",
    "high_constraint": "#ff7f0e",
    "loose": "#2ca02c",
}

draw_order = ["balanced", "high_constraint", "loose"]

# Tiny horizontal jitter (in size units) to prevent perfect overlap when values are equal
# Keeps the x-axis ticks exactly at the required sizes.
x_jitter = {"balanced": -3.0, "high_constraint": 0.0, "loose": 3.0}

def _ensure_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        if s.dropna().max() <= 1.2:
            out[c] = s * 100.0
    return out

def main() -> None:
    df = pd.read_csv(csv_path)
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )
    df = df[df[size_col].isin(x_sizes)].copy()
    df = _ensure_percent(df, score_cols)
    df.sort_values(["grammar_canon", size_col], inplace=True)

    means_by_corpus = df.groupby("grammar_canon", as_index=False)[score_cols].mean()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    top_axes, bottom_axes = axes[0], axes[1]
    score_titles = ["Score = 0", "Score = 1", "Score = 2"]

    # Row 1: line charts (impact of size) with jitter to avoid hiding overlaps
    for j, (col, title) in enumerate(zip(score_cols, score_titles)):
        ax = top_axes[j]
        for corpus in draw_order:
            sub = df[df["grammar_canon"] == corpus]
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
                solid_capstyle="round",
            )
        ax.set_title(title)
        ax.set_xlabel("Corpus size")
        ax.set_ylabel("Percentage of answers")
        ax.set_xticks(x_sizes)  # ticks remain at the true sizes
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(True, linestyle="--", alpha=0.3)

    # One legend for the top row
    fig.legend(
        [plt.Line2D([0], [0], color=colour_map[c], lw=3) for c in draw_order],
        draw_order,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    # Row 2: bar charts (impact of corpus, averaged across sizes)
    for j, (col, title) in enumerate(zip(score_cols, score_titles)):
        ax = bottom_axes[j]
        heights = [
            means_by_corpus.loc[means_by_corpus["grammar_canon"] == c, col].squeeze()
            if c in means_by_corpus["grammar_canon"].values else 0.0
            for c in draw_order
        ]
        ax.bar(
            range(len(draw_order)),
            heights,
            tick_label=draw_order,
            color=[colour_map[c] for c in draw_order],
        )
        ax.set_title(title)
        ax.set_xlabel("Corpus type")
        ax.set_ylabel("Percentage of answers")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Score Distribution: Impact of Size (Top) and Corpus Type (Bottom)", y=1.06, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
