"""
Produce:
1) 'completeness_gain_and_c_resp_vs_size.png' — two side-by-side line plots:
   - Average Completeness Gain vs Corpus Size
   - Average c_resp vs Corpus Size
2) 'c_resp_vs_c_base_by_corpus.png' — grouped bar plot comparing average c_resp vs c_base
   per corpus (averaged across sizes), with c_base drawn lighter.

Conventions:
- X-axis (lines): corpus sizes [50, 100, 200, 300, 400, 500]
- Y-axis: metric score
- Colours (exact Tab10): balanced=#1f77b4, high_constraint=#ff7f0e, loose=#2ca02c
- Lines: thick with clear markers
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ---- fixed inputs ------------------------------------------------------------

csv_path = Path("completeness_gain_summary_20250916_140751.csv")
output_path_lines = Path("completeness_gain_and_c_resp_vs_size.png")
output_path_bars  = Path("c_resp_vs_c_base_by_corpus.png")

grammar_col      = "grammar"
size_col         = "size"
metric_gain_col  = "completeness_gain"
metric_resp_col  = "c_resp"
metric_base_col  = "c_base"

# Sizes to display on the x-axis (others will be ignored)
x_sizes = [50, 100, 200, 300, 400, 500]

# Canonical mapping for grammar values (normalises any variants)
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Fixed colour mapping (must never change) — exact Tab10 hexes
colour_map = {
    "balanced": "#1f77b4",        # tab:blue
    "high_constraint": "#ff7f0e", # tab:orange
    "loose": "#2ca02c",           # tab:green
}
draw_order = ["balanced", "high_constraint", "loose"]

# Tiny horizontal jitter to avoid perfectly overlapping markers/lines
x_jitter = {"balanced": -3.0, "high_constraint": 0.0, "loose": 3.0}

# ---- styling knobs (thicker lines & bigger axis numbers) ---------------------
LINE_WIDTH = 5.0        # thicker curves
MARKER_SIZE = 10        # larger markers
TITLE_SIZE = 18         # subplot title size
AXIS_LABEL_SIZE = 16    # x/y label size
TICK_LABEL_SIZE = 14    # numbers on the axes
LEGEND_FONT_SIZE = 14   # legend text size


def main() -> None:
    # ---- load & tidy ---------------------------------------------------------
    df = pd.read_csv(csv_path)
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )
    df = df[df[size_col].isin(x_sizes)].copy()

    # Aggregate means by (grammar, size) in case there are multiple runs
    agg = (
        df.groupby(["grammar_canon", size_col], as_index=False)[
            [metric_gain_col, metric_resp_col, metric_base_col]
        ].mean()
    )

    # ---- Figure 1: two line plots -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Average Completeness Gain vs Size
    ax = axes[0]
    for corpus in draw_order:
        sub = agg[agg["grammar_canon"] == corpus]
        if sub.empty:
            continue
        x = sub[size_col].astype(float) + x_jitter[corpus]
        ax.plot(
            x,
            sub[metric_gain_col],
            label=corpus,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            c=colour_map[corpus],
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
    ax.set_title("Average Completeness Gain vs Corpus Size", fontsize=TITLE_SIZE)
    ax.set_xlabel("Corpus size", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Average Completeness Gain", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_sizes)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Right: Average c_resp vs Size
    ax = axes[1]
    for corpus in draw_order:
        sub = agg[agg["grammar_canon"] == corpus]
        if sub.empty:
            continue
        x = sub[size_col].astype(float) + x_jitter[corpus]
        ax.plot(
            x,
            sub[metric_resp_col],
            label=corpus,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            c=colour_map[corpus],
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
    ax.set_title("Average c_resp vs Corpus Size", fontsize=TITLE_SIZE)
    ax.set_xlabel("Corpus size", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Average c_resp", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_sizes)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Single legend for figure 1 — place BELOW the plots so it doesn't cover titles
    handles = [plt.Line2D([0], [0], color=colour_map[c], lw=LINE_WIDTH) for c in draw_order]
    labels = draw_order
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        prop={"size": LEGEND_FONT_SIZE},
    )
    # Make room for the bottom legend
    fig.subplots_adjust(bottom=0.20)

    plt.tight_layout()
    plt.savefig(output_path_lines, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path_lines.resolve()}")

    # ---- Figure 2: grouped bar plot (avg over sizes) ------------------------
    means_by_corpus = (
        df.groupby("grammar_canon", as_index=False)[[metric_resp_col, metric_base_col]].mean()
    )

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    index = list(range(len(draw_order)))
    bar_width = 0.38

    for i, corpus in enumerate(draw_order):
        present = set(means_by_corpus["grammar_canon"].values)
        val_resp = float(means_by_corpus.loc[means_by_corpus["grammar_canon"] == corpus, metric_resp_col].squeeze()) if corpus in present else 0.0
        val_base = float(means_by_corpus.loc[means_by_corpus["grammar_canon"] == corpus, metric_base_col].squeeze()) if corpus in present else 0.0

        # c_resp: solid corpus colour with hatch
        ax2.bar(i - bar_width/2, val_resp, width=bar_width,
                color=colour_map[corpus], hatch="//", edgecolor="black", linewidth=0.7, label=None)

        # c_base: same corpus colour but lighter (alpha) + different hatch
        ax2.bar(i + bar_width/2, val_base, width=bar_width,
                color=colour_map[corpus], alpha=0.45, hatch="..",
                edgecolor="black", linewidth=0.7, label=None)

    ax2.set_xticks(index)
    ax2.set_xticklabels(draw_order, fontsize=TICK_LABEL_SIZE)
    ax2.set_ylabel("Average score", fontsize=AXIS_LABEL_SIZE)
    ax2.set_title("Average RAG Raw Completeness (c_resp) vs Baseline (c_base) by Corpus (Averaged over Sizes)",
                  fontsize=TITLE_SIZE)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    # Legend proxies to show hatches and the lighter c_base
    proxy_resp = plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", ec="black", hatch="//", label="c_resp")
    proxy_base = plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", ec="black", hatch="..", alpha=0.45, label="c_base")
    ax2.legend(handles=[proxy_resp, proxy_base], frameon=True, loc="upper left", prop={"size": LEGEND_FONT_SIZE})

    plt.tight_layout()
    plt.savefig(output_path_bars, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figure to: {output_path_bars.resolve()}")


if __name__ == "__main__":
    main()
