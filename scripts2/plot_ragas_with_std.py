# scripts2/plot_ragas_with_std.py
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"grammar", "size", "faithfulness", "answer_relevancy"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce")
    df["answer_relevancy"] = pd.to_numeric(df["answer_relevancy"], errors="coerce")
    df = df.dropna(subset=["size", "faithfulness", "answer_relevancy"])
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["grammar", "size"])
          .agg(
              faithfulness_mean=("faithfulness", "mean"),
              faithfulness_std=("faithfulness", "std"),
              answer_relevancy_mean=("answer_relevancy", "mean"),
              answer_relevancy_std=("answer_relevancy", "std"),
              n_items=("grammar", "count"),
          )
          .reset_index()
          .sort_values(["grammar", "size"])
    )
    agg["faithfulness_std"] = agg["faithfulness_std"].fillna(0.0)
    agg["answer_relevancy_std"] = agg["answer_relevancy_std"].fillna(0.0)
    return agg


def make_color_map(grammars: list[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(grammars))}


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def plot_with_band(agg: pd.DataFrame, metric_mean: str, metric_std: str, title: str, out_path: Path):
    grammars = sorted(agg["grammar"].unique().tolist())
    cmap = make_color_map(grammars)

    plt.figure()
    for g in grammars:
        sub = agg[agg["grammar"] == g].sort_values("size")
        x = sub["size"].to_numpy()
        y = sub[metric_mean].to_numpy()
        s = sub[metric_std].to_numpy()
        y_low = _clip01(y - s)
        y_high = _clip01(y + s)

        plt.plot(x, y, marker="o", linewidth=1.8, label=g, color=cmap[g])
        plt.fill_between(x, y_low, y_high, color=cmap[g], alpha=0.18)

    plt.xlabel("Corpus size")
    ylabel = metric_mean.replace("_", " ").title() + " Mean"
    plt.ylabel(ylabel)
    plt.title(f"{title} (±1 SD)")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def main():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Plot RAGAS (mean ± SD) with shaded bands and export aggregated CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Per-item RAGAS CSV (from eval script).")
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/ragas_agg_with_std.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/plots"))
    args = parser.parse_args()

    df = load_data(args.csv)
    agg = aggregate(df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_csv, index=False)

    plot_with_band(
        agg,
        metric_mean="faithfulness_mean",
        metric_std="faithfulness_std",
        title="RAGAS Faithfulness vs Corpus Size",
        out_path=args.out_dir / "faithfulness_vs_size_band.png",
    )
    plot_with_band(
        agg,
        metric_mean="answer_relevancy_mean",
        metric_std="answer_relevancy_std",
        title="RAGAS Answer Relevancy vs Corpus Size",
        out_path=args.out_dir / "answer_relevancy_vs_size_band.png",
    )


if __name__ == "__main__":
    main()
