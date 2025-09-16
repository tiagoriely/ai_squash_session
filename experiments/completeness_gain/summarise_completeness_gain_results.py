"""
Summarise the completeness visuals into CSVs so you can quote the exact values shown.

Reads:
  - 'completeness_gain_summary_20250916_140751.csv'

Writes (three CSVs with exactly the values used in the figures):
  1) 'summary_completeness_gain_by_size.csv'
     Columns: grammar, size, avg_completeness_gain
     (Means per grammar × size; used for the "Average Completeness Gain vs Corpus Size" line plot)

  2) 'summary_c_resp_by_size.csv'
     Columns: grammar, size, avg_c_resp
     (Means per grammar × size; used for the "Average c_resp vs Corpus Size" line plot)

  3) 'summary_c_resp_vs_c_base_by_corpus.csv'
     Columns: grammar, avg_c_resp, avg_c_base
     (Means per grammar across sizes; used for the grouped bar plot comparing c_resp vs c_base)

Conventions kept consistent with your visuals:
- X-axis sizes restricted to [100, 200, 300, 400, 500]
- Grammar labels normalised: 'high' -> 'high_constraint'
- Corpus order: balanced, high_constraint, loose
"""

from pathlib import Path
import sys
import pandas as pd


# ---- fixed inputs/outputs ----------------------------------------------------

input_path = Path("completeness_gain_summary_20250916_140751.csv")

out_gain_by_size  = Path("summary_completeness_gain_by_size.csv")
out_cresp_by_size = Path("summary_c_resp_by_size.csv")
out_bars_by_corp  = Path("summary_c_resp_vs_c_base_by_corpus.csv")

# Expected column names in the source CSV
grammar_col = "grammar"
size_col    = "size"
gain_col    = "completeness_gain"
c_resp_col  = "c_resp"
c_base_col  = "c_base"

# Sizes that appear on the plots
x_sizes = [50, 100, 200, 300, 400, 500]

# Canonical mapping for grammar values (British English comments)
canonical = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

# Stable output ordering
corpus_order = ["balanced", "high_constraint", "loose"]


def fail(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Validate columns
    needed = {grammar_col, size_col, gain_col, c_resp_col, c_base_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        fail(f"Missing required column(s): {', '.join(missing)}")

    # Normalise grammar labels to canonical ones
    df["grammar_canon"] = (
        df[grammar_col].astype(str).str.lower().map(canonical).fillna(df[grammar_col])
    )

    # Keep only the sizes that are on the figures
    df = df[df[size_col].isin(x_sizes)].copy()

    # ---- 1) Average Completeness Gain vs Size (exact line values) ------------
    gain_by_size = (
        df.groupby(["grammar_canon", size_col], as_index=False)[gain_col]
          .mean()
          .rename(columns={
              "grammar_canon": "grammar",
              gain_col: "avg_completeness_gain"
          })
    )
    # Order rows as on the plot
    gain_by_size["grammar"] = pd.Categorical(gain_by_size["grammar"], categories=corpus_order, ordered=True)
    gain_by_size = gain_by_size.sort_values(["grammar", size_col])

    gain_by_size.to_csv(out_gain_by_size, index=False)
    print(f"Written: {out_gain_by_size.resolve()}")

    # ---- 2) Average c_resp vs Size (exact line values) -----------------------
    c_resp_by_size = (
        df.groupby(["grammar_canon", size_col], as_index=False)[c_resp_col]
          .mean()
          .rename(columns={
              "grammar_canon": "grammar",
              c_resp_col: "avg_c_resp"
          })
    )
    c_resp_by_size["grammar"] = pd.Categorical(c_resp_by_size["grammar"], categories=corpus_order, ordered=True)
    c_resp_by_size = c_resp_by_size.sort_values(["grammar", size_col])

    c_resp_by_size.to_csv(out_cresp_by_size, index=False)
    print(f"Written: {out_cresp_by_size.resolve()}")

    # ---- 3) Average c_resp vs c_base per corpus (exact bar values) -----------
    bars_by_corpus = (
        df.groupby("grammar_canon", as_index=False)[[c_resp_col, c_base_col]]
          .mean()
          .rename(columns={
              "grammar_canon": "grammar",
              c_resp_col: "avg_c_resp",
              c_base_col: "avg_c_base",
          })
    )
    bars_by_corpus["grammar"] = pd.Categorical(bars_by_corpus["grammar"], categories=corpus_order, ordered=True)
    bars_by_corpus = bars_by_corpus.sort_values(["grammar"])

    bars_by_corpus.to_csv(out_bars_by_corp, index=False)
    print(f"Written: {out_bars_by_corp.resolve()}")

    print("\nDone. These CSVs contain exactly the values used in the corresponding visuals.")


if __name__ == "__main__":
    main()
