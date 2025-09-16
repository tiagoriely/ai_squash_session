"""
Summarise RAGAS visuals (Answer Relevancy & Faithfulness) into CSVs so you can
quote the exact values shown on your plots.

Reads
-----
- 'ragas_scores_all_v1.csv'

Writes (values are means per grammar × size; includes count used)
-----------------------------------------------------------------
1) 'ragas_answer_relevancy_by_size.csv'
   Columns: grammar, size, avg_answer_relevancy, n

2) 'ragas_faithfulness_by_size.csv'
   Columns: grammar, size, avg_faithfulness, n

Notes
-----
- Grammar labels are normalised so any 'high' becomes 'high_constraint'.
- Sizes are restricted to the standard x-axis ticks: [100, 200, 300, 400, 500].
  (If you also want size=50 included, flip INCLUDE_SIZE_50 to True below.)
"""

from pathlib import Path
import sys
import pandas as pd


# -------- fixed inputs/outputs -----------------------------------------------

CSV_IN  = Path("ragas_scores_all_v1.csv")

OUT_RELEVANCY   = Path("ragas_answer_relevancy_by_size.csv")
OUT_FAITHFULNESS = Path("ragas_faithfulness_by_size.csv")

# Standard sizes for your visuals
X_SIZES = [50, 100, 200, 300, 400, 500]

# Optionally include size=50 as well (set True if your plot included it)
INCLUDE_SIZE_50 = False


# -------- column names in the input ------------------------------------------

COL_GRAMMAR = "grammar"
COL_SIZE = "size"
COL_RELEVANCY = "answer_relevancy"
COL_FAITHFULNESS = "faithfulness"


# -------- helpers -------------------------------------------------------------

CANONICAL = {
    "balanced": "balanced",
    "high": "high_constraint",
    "high_constraint": "high_constraint",
    "loose": "loose",
}

ORDER = ["balanced", "high_constraint", "loose"]


def fail(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not CSV_IN.exists():
        fail(f"Input file not found: {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    # Validate columns
    needed = {COL_GRAMMAR, COL_SIZE, COL_RELEVANCY, COL_FAITHFULNESS}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        fail(f"Missing required column(s): {', '.join(missing)}")

    # Normalise grammar labels
    df["grammar_canon"] = (
        df[COL_GRAMMAR].astype(str).str.lower().map(CANONICAL).fillna(df[COL_GRAMMAR])
    )

    # Restrict to the sizes used on the visuals (optionally include 50)
    sizes = X_SIZES if not INCLUDE_SIZE_50 else sorted(set(X_SIZES + [50]))
    df = df[df[COL_SIZE].isin(sizes)].copy()

    # ---- Answer Relevancy summary -------------------------------------------
    rel = (
        df.groupby(["grammar_canon", COL_SIZE], as_index=False)
          .agg(avg_answer_relevancy=(COL_RELEVANCY, "mean"), n=(COL_RELEVANCY, "size"))
          .rename(columns={"grammar_canon": "grammar"})
    )
    # Stable ordering
    rel["grammar"] = pd.Categorical(rel["grammar"], categories=ORDER, ordered=True)
    rel = rel.sort_values(["grammar", COL_SIZE])

    rel.to_csv(OUT_RELEVANCY, index=False)
    print(f"Wrote: {OUT_RELEVANCY.resolve()}")

    # ---- Faithfulness summary ------------------------------------------------
    faith = (
        df.groupby(["grammar_canon", COL_SIZE], as_index=False)
          .agg(avg_faithfulness=(COL_FAITHFULNESS, "mean"), n=(COL_FAITHFULNESS, "size"))
          .rename(columns={"grammar_canon": "grammar"})
    )
    faith["grammar"] = pd.Categorical(faith["grammar"], categories=ORDER, ordered=True)
    faith = faith.sort_values(["grammar", COL_SIZE])

    faith.to_csv(OUT_FAITHFULNESS, index=False)
    print(f"Wrote: {OUT_FAITHFULNESS.resolve()}")

    print("\nDone. These CSVs contain the exact values used for the curves.")


if __name__ == "__main__":
    main()
