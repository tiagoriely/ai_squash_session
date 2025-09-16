# comparative_performance_plots.py
# Usage:
#   python comparative_performance_plots.py \
#     --input hybrid_retrievers_metrics.csv \
#     --outdir plots

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------

def ensure_dir(p: Path):
    """Create a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def natural_sort_complex_ids(ids):
    """Sort complex query IDs in a natural order (e.g., complex_1_cg, complex_2_cg)."""

    def key(x):
        m = re.search(r"complex_(\d+)_cg", str(x))
        return int(m.group(1)) if m else 10 ** 9

    return sorted(ids, key=key)


def natural_sort_complex_mix_ids(ids):
    """Sorts 'mix' query IDs in natural order (e.g., complex_21_mix, complex_22_mix)."""

    def key(x):
        m = re.search(r"complex_(\d+)_mix", str(x))
        return int(m.group(1)) if m else 10 ** 9

    return sorted(ids, key=key)


def normalise_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the max_score for each retriever strategy to the [0, 1] range,
    except for 'dynamic_query_aware_score_fusion' which is already normalised.
    """
    strategies_to_normalise = [
        s for s in df['strategy_name'].unique()
        if s != 'dynamic_query_aware_score_fusion'
    ]
    df_normalised = df.copy()
    for strategy in strategies_to_normalise:
        strategy_mask = df_normalised['strategy_name'] == strategy
        scores = df_normalised.loc[strategy_mask, 'max_score']
        min_score = scores.min()
        max_score = scores.max()
        score_range = max_score - min_score
        if score_range > 0:
            df_normalised.loc[strategy_mask, 'max_score'] = (scores - min_score) / score_range
        else:
            df_normalised.loc[strategy_mask, 'max_score'] = 0.0
    return df_normalised


# ---------- plotting ----------

def plot_avg_max_by_qtype(df: pd.DataFrame, outdir: Path):
    """Generates a grouped bar chart with distinct colors and hatch patterns for query types."""
    ensure_dir(outdir)
    g = (df.groupby(["strategy_name", "query_type"])["max_score"]
         .mean()
         .unstack(fill_value=0.0)
         .sort_index())

    strategies = list(g.index)
    qtypes = list(g.columns)

    # ** MODIFICATION: Define distinct colors AND hatch patterns **
    palette = plt.get_cmap('tab10').colors
    hatch_patterns = ['/', '\\', 'x', '.', '*', '+', 'O', '|', '-']

    color_map_qtype = {qt: palette[i % len(palette)] for i, qt in enumerate(qtypes)}
    hatch_map_qtype = {qt: hatch_patterns[i % len(hatch_patterns)] for i, qt in enumerate(qtypes)}

    x = np.arange(len(strategies))
    width = 0.8 / max(1, len(qtypes))
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, qt in enumerate(qtypes):
        color = color_map_qtype.get(qt)
        hatch = hatch_map_qtype.get(qt)  # Get the specific hatch pattern
        ax.bar(x + i * width - (width * len(qtypes) / 2) + width / 2,
               g[qt].values,
               width,
               label=str(qt),
               color=color,
               hatch=hatch,  # Apply the hatch pattern
               edgecolor='white'  # Add an edge to make hatches clearer
               )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylabel("Average Maximum Score")
    ax.set_xlabel("Retriever (strategy)")
    ax.set_title("Average Maximum Score by Query Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Query type", frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(outdir / "avg_max_score_by_query_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_avg_relative_gap(df: pd.DataFrame, outdir: Path):
    """
    Generates a bar chart for the average relative confidence gap
    with consistent colouring based on retriever strategy.
    """
    ensure_dir(outdir)

    color_map_strategy = {
        'static_weighted_rrf': "#1f77b4",
        'standard_unweighted_rrf': "#ff7f0e",
        'dynamic_query_aware_rrf': "#2ca02c",
        'dynamic_query_aware_score_fusion': "#d62728"
    }

    df['relative_gap'] = (df['top_1_delta'] / df['max_score']).fillna(0)

    s = (df.groupby("strategy_name")["relative_gap"]
         .mean()
         .sort_values(ascending=False))

    bar_colors = [color_map_strategy.get(strategy, '#808080') for strategy in s.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(s.index, s.values, color=bar_colors)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylabel("Average Relative Gap (% Score Drop)")
    ax.set_xlabel("Retriever (strategy)")
    ax.set_title("Average Relative Confidence Gap (Rank 1 vs Rank 2)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(s.index, rotation=15, ha="right")

    for i, v in enumerate(s.values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(outdir / "average_relative_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_performance(df: pd.DataFrame, outdir: Path):
    """
    Generates a line plot for Complexity Type 1 queries, with a horizontal offset (jitter).
    """
    ensure_dir(outdir)
    cg = df[df["query_id"].str.contains(r"^complex_\d+_cg$", regex=True)].copy()
    if cg.empty:
        print("Warning: No complexity type 1 queries found. Skipping this plot.")
        return

    perf = (cg.pivot_table(index="strategy_name", columns="query_id", values="max_score", aggfunc="first"))
    cols_order = natural_sort_complex_ids(perf.columns.tolist())
    perf = perf.reindex(columns=cols_order).sort_index()

    x = np.arange(len(cols_order))
    fig, ax = plt.subplots(figsize=(10, 6))
    num_strategies = len(perf.index)
    jitter_width = 0.08
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, num_strategies)
    strategy_offsets = dict(zip(perf.index, offsets))

    for strat, row in perf.iterrows():
        offset = strategy_offsets[strat]
        ax.plot(x + offset, row.values.astype(float), marker="o", linestyle="-", label=strat, linewidth=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cols_order, rotation=30, ha="right")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Top-1 Score (maximum per query)")
    ax.set_title("Performance on Graduated Complexity Queries (Type 1)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Retriever", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "performance_on_graduated_complexity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_performance_type2(df: pd.DataFrame, outdir: Path):
    """
    Generates a line plot for Complexity Type 2 queries, with a horizontal offset (jitter).
    """
    ensure_dir(outdir)
    mix = df[df["query_id"].str.contains(r"^complex_2\d_mix$", regex=True)].copy()
    if mix.empty:
        print("Warning: No complexity type 2 queries found. Skipping this plot.")
        return

    perf = (mix.pivot_table(index="strategy_name", columns="query_id", values="max_score", aggfunc="first"))
    cols_order = natural_sort_complex_mix_ids(perf.columns.tolist())
    perf = perf.reindex(columns=cols_order).sort_index()

    x = np.arange(len(cols_order))
    fig, ax = plt.subplots(figsize=(10, 6))
    num_strategies = len(perf.index)
    jitter_width = 0.08
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, num_strategies)
    strategy_offsets = dict(zip(perf.index, offsets))

    for strat, row in perf.iterrows():
        offset = strategy_offsets[strat]
        ax.plot(x + offset, row.values.astype(float), marker="o", linestyle="-", label=strat, linewidth=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cols_order, rotation=30, ha="right")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Top-1 Score (maximum per query)")
    ax.set_title("Performance on Graduated Complexity Queries (Type 2 - Mix)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Retriever", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "performance_on_graduated_complexity_type2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main():
    """Main function to parse arguments, load data, normalise scores, and generate plots."""
    ap = argparse.ArgumentParser(description="Generate comparative performance plots for hybrid retrievers.")
    ap.add_argument("--input", default="hybrid_retrievers_metrics.csv",
                    help="Path to the input metrics CSV file.")
    ap.add_argument("--outdir", default="plots", help="Directory to save the plot images.")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)

    if not inp.exists():
        print(f"Warning: Input file '{inp}' not found. Creating a dummy file for demonstration.")
        data, strategies = [], ['static_weighted_rrf', 'standard_unweighted_rrf', 'dynamic_query_aware_rrf',
                                'dynamic_query_aware_score_fusion']
        queries = [{"query_id": f"complex_{i:02d}_{suffix}", "type": f"Complexity Type {t} (Relevant)"} for
                   t, suffix, count in [(1, 'cg', 3), (2, 'mix', 4)] for i in range(1, count + 1)]
        np.random.seed(42)
        for strategy in strategies:
            for query in queries:
                is_relevant = 'Relevant' in query["type"]
                if query["query_id"] == "complex_01_cg" and strategy != 'dynamic_query_aware_score_fusion':
                    max_score = 0.025
                elif strategy == 'dynamic_query_aware_score_fusion':
                    score_map = {"complex_01_cg": 0.68, "complex_02_cg": 0.81, "complex_03_cg": 0.95,
                                 "complex_21_mix": 0.70, "complex_22_mix": 0.75, "complex_23_mix": 0.85,
                                 "complex_24_mix": 0.90}
                    max_score = score_map.get(query["query_id"], np.random.uniform(0.6, 0.8))
                else:
                    max_score = np.random.uniform(0.015, 0.025) if is_relevant else np.random.uniform(0.001, 0.01)
                data.append({'strategy_name': strategy, 'query_id': query["query_id"], 'query_type': query["type"],
                             'max_score': max_score, 'top_1_delta': max_score * np.random.uniform(0.1, 0.4)})
        pd.DataFrame(data).to_csv(inp, index=False)

    df = pd.read_csv(inp)
    df_normalised = normalise_scores(df)

    plot_avg_max_by_qtype(df_normalised, outdir)
    plot_avg_relative_gap(df, outdir)
    plot_complexity_performance(df_normalised, outdir)
    plot_complexity_performance_type2(df_normalised, outdir)

    print(f"\nPlots have been successfully generated in the '{outdir}' directory.")


if __name__ == "__main__":
    main()