# scripts2/eval_ragas_from_json.py
import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# env & path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# ragas wrapper
try:
    from evaluation.utils.ragas import RagasEvaluator
except Exception as _e:
    RagasEvaluator = None


# ----------------------- helpers -----------------------

def resolve_input_paths(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            matched = [Path(x) for x in glob(str(p))]
            if not matched and p.suffix.lower() == ".jso":
                alt = Path(str(p) + "n")
                if alt.exists():
                    matched = [alt]
            out.extend(matched if matched else ([p] if p.exists() else []))
    seen = set()
    uniq = []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def parse_grammar_and_size_from_case_id(case_id: str):
    parts = case_id.split("_")
    if not parts:
        return None, None
    size_idx = None
    for i, tok in enumerate(parts):
        if tok.isdigit():
            size_idx = i
            break
    if size_idx is None:
        return "_".join(parts), None
    grammar = "_".join(parts[:size_idx])
    size = int(parts[size_idx])
    return grammar, size


def try_parse_size_from_filename(path: Path) -> int | None:
    name = path.name
    m = re.search(r"_size(\d+)_", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_(\d{2,7})_", name)
    if m2:
        return int(m2.group(1))
    return None


def load_corpus_index(grammar: str, size: int) -> dict[str, str]:
    corpus_path = PROJECT_ROOT / f"data/processed/{grammar}_grammar/{grammar}_{size}.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    idx: dict[str, str] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                did = doc.get("id")
                contents = doc.get("contents", "")
                if did:
                    idx[did] = contents
            except Exception:
                continue
    return idx


def build_context(corpus_index: dict[str, str], retrieved_ids: list[str], top_k: int) -> str:
    if not isinstance(retrieved_ids, list):
        retrieved_ids = []
    parts = []
    for did in retrieved_ids[:top_k]:
        if did in corpus_index:
            parts.append(f"Source Document ID: {did}\n\n{corpus_index[did]}")
        else:
            parts.append(f"Source Document ID: {did}\n\n")
    return "\n\n---\n\n".join(parts)


def make_ragas_evaluator():
    if RagasEvaluator is None:
        raise ImportError(
            "RagasEvaluator not found. Provide evaluation.utils.ragas.RagasEvaluator with .evaluate(query, context, generated_plan)."
        )
    return RagasEvaluator()


def plot_metric_vs_size(agg_df: pd.DataFrame, metric: str, out_path: Path, title: str):
    plt.figure()
    for grammar, g in agg_df.groupby("grammar"):
        g = g.sort_values("size")
        plt.plot(g["size"], g[metric], marker="o", label=grammar)
    plt.xlabel("Corpus size")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)


# ----------------------- main eval -----------------------

def evaluate_files(
    input_paths: list[Path],
    top_k_context: int = 10,
    show_per_item: bool = False,
    by_query: bool = False,
    out_csv: Path | None = None,
    out_plot_dir: Path | None = None,
):
    frames = []
    for p in input_paths:
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping unreadable file: {p} ({e})")
            continue
        if not isinstance(items, list) or not items:
            print(f"Skipping empty/non-list file: {p}")
            continue
        df = pd.DataFrame(items)
        df["__source_file"] = str(p)
        frames.append(df)

    if not frames:
        print("No valid JSON items to evaluate.")
        return

    df = pd.concat(frames, ignore_index=True)

    required = {"case_id", "query_text", "retrieved_documents_info", "generated_plan"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields in inputs: {missing}")

    parsed = df["case_id"].apply(parse_grammar_and_size_from_case_id)
    df["grammar"] = [g for (g, s) in parsed]
    df["size"] = [s for (g, s) in parsed]

    mask_missing = df["size"].isna()
    if mask_missing.any():
        for idx in df[mask_missing].index:
            guess = try_parse_size_from_filename(Path(df.at[idx, "__source_file"]))
            df.at[idx, "size"] = guess

    unresolved = df[df["grammar"].isna() | df["size"].isna()]
    if not unresolved.empty:
        print(f"Dropping {len(unresolved)} rows with unresolved grammar/size.")
        df = df.drop(index=unresolved.index).reset_index(drop=True)

    if df.empty:
        print("No rows left after filtering unresolved grammar/size.")
        return

    inv = df.groupby(["grammar", "size"]).size().reset_index(name="n_items")
    print("\nInventory by grammar & size:")
    print(inv.to_string(index=False))

    ragas = make_ragas_evaluator()
    corpus_cache: dict[tuple[str, int], dict[str, str]] = {}

    rows = []
    print("\nScoring with RAGAS (faithfulness, answer_relevancy)...\n")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        grammar = str(row["grammar"])
        size = int(row["size"])
        query = row["query_text"]
        plan = row["generated_plan"]
        retrieved_ids = row.get("retrieved_documents_info", []) or []

        key = (grammar, size)
        if key not in corpus_cache:
            corpus_cache[key] = load_corpus_index(grammar, size)

        context_str = build_context(corpus_cache[key], retrieved_ids, top_k_context)
        try:
            scores = ragas.evaluate(query=query, context=context_str, generated_plan=plan)
            faith = float(scores.get("faithfulness", 0.0))
            ansrel = float(scores.get("answer_relevancy", 0.0))
        except Exception as e:
            print(f"RAGAS failed on case_id={row['case_id']}: {e}")
            faith, ansrel = 0.0, 0.0

        rows.append(
            {
                "source_file": row["__source_file"],
                "case_id": row["case_id"],
                "grammar": grammar,
                "size": size,
                "query_text": query,
                "faithfulness": faith,
                "answer_relevancy": ansrel,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        print("No scores produced.")
        return

    if show_per_item:
        print("\nPer-item RAGAS scores (first 100 rows):")
        print(results.head(100).to_string(index=False))

    print("\n=== AVERAGES BY GRAMMAR ===")
    by_grammar = results.groupby("grammar")[["faithfulness", "answer_relevancy"]].mean().round(4)
    print(by_grammar.to_string())

    print("\n=== AVERAGES BY GRAMMAR & SIZE ===")
    by_gs = results.groupby(["grammar", "size"])[["faithfulness", "answer_relevancy"]].mean().reset_index().sort_values(["grammar", "size"])
    print(by_gs.to_string(index=False))

    if by_query:
        print("\n=== AVERAGES BY GRAMMAR & QUERY ===")
        by_gq = results.groupby(["grammar", "query_text"])[["faithfulness", "answer_relevancy"]].mean().round(4)
        if len(by_gq) > 120:
            print(by_gq.head(120).to_string())
            print(f"... ({len(by_gq)} rows total)")
        else:
            print(by_gq.to_string())

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_csv, index=False)
        print(f"\nSaved per-item scores to: {out_csv}")

    if out_plot_dir:
        out_plot_dir = Path(out_plot_dir)
        plot_metric_vs_size(by_gs, "faithfulness", out_plot_dir / "faithfulness_vs_size.png", "RAGAS Faithfulness vs Corpus Size")
        plot_metric_vs_size(by_gs, "answer_relevancy", out_plot_dir / "answer_relevancy_vs_size.png", "RAGAS Answer Relevancy vs Corpus Size")
        print(f"Saved plots to: {out_plot_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RAGAS faithfulness & answer relevancy from existing JSONs; grouped by grammar and size, with plots."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Files/dirs/globs of generated-session JSONs (no generation or retrieval will be run).",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k context docs from retrieved_documents_info.")
    parser.add_argument("--show-per-item", action="store_true", help="Print per-item rows (first 100).")
    parser.add_argument("--by-query", action="store_true", help="Also show averages by grammar & query.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for per-item scores.")
    parser.add_argument("--plot-dir", type=Path, default=PROJECT_ROOT / "experiments" / "plots",
                        help="Directory to save plots (faithfulness_vs_size.png, answer_relevancy_vs_size.png).")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    args = parser.parse_args()

    paths = resolve_input_paths(args.inputs)
    evaluate_files(
        input_paths=paths,
        top_k_context=args.k,
        show_per_item=args.show_per_item,
        by_query=args.by_query,
        out_csv=args.csv,
        out_plot_dir=None if args.no_plot else args.plot_dir,
    )


if __name__ == "__main__":
    main()
