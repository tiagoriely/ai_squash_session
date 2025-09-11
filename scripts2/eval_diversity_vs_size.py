# scripts2/eval_diversity_vs_size.py
import argparse, json, hashlib, itertools, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from evaluation.evaluators.llm_judge import LlmEvaluator  # only used if we need to extract signatures

# ---------- helpers ----------
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_grammar_size(case_id: str):
    # expects like: "{grammar}_{size}_{queryid}_k{K}"
    parts = case_id.split("_")
    grammar = parts[0]
    # find first int-looking token as size
    size = None
    for p in parts[1:]:
        if p.isdigit():
            size = int(p); break
    return grammar, size

def signature_from_embedded(plan_text: str):
    """
    If your generation already appends a JSON block called STRUCTURAL_SIGNATURE: {...}
    try to parse it here. Return dict or None if not present.
    """
    import re, json as _json
    m = re.search(r"STRUCTURAL_SIGNATURE\s*:\s*({.*?})", plan_text, flags=re.S|re.I)
    if not m:
        return None
    try:
        return _json.loads(m.group(1))
    except Exception:
        return None

def compute_pair_diffs(sigA: dict, sigB: dict) -> dict:
    """Deterministic facet diffs from two signatures."""
    # normalize helpers
    def _set_ex(sig):
        ex = sig.get("exercises", []) or []
        return set((e.get("family","").strip().lower(),
                    e.get("variant","").strip().lower(),
                    e.get("side","").strip().lower()) for e in ex)
    def _set_rules(sig):
        return set([r.strip().lower() for r in sig.get("rules",[]) or []])
    def _set_motifs(sig):
        # motifs is list of lists; convert each to tuple of tokens
        mot = sig.get("motifs",[]) or []
        norm = set(tuple(tok.strip().lower() for tok in m) for m in mot if isinstance(m, (list, tuple)))
        return norm
    def _blocks(sig):
        # list of dicts with {name, minutes, type}; we compare the sequence of "type" and "name"
        bl = sig.get("blocks",[]) or []
        seq = [ ( (b.get("name","") or "").strip().lower(), (b.get("type","") or "").strip().lower() ) for b in bl ]
        return seq
    def _focus(sig):
        return set([f.strip().lower() for f in sig.get("focus",[]) or []])
    def _side_balance(sig):
        # coarse: if exercises include both sides or one dominant side
        ex = sig.get("exercises",[]) or []
        sides = [ (e.get("side","") or "").strip().lower() for e in ex if e.get("side") ]
        return "both" if ("forehand" in sides and "backhand" in sides) else (sides[0] if sides else "unknown")

    exA, exB = _set_ex(sigA), _set_ex(sigB)
    rulesA, rulesB = _set_rules(sigA), _set_rules(sigB)
    motA, motB = _set_motifs(sigA), _set_motifs(sigB)
    blkA, blkB = _blocks(sigA), _blocks(sigB)
    focA, focB = _focus(sigA), _focus(sigB)
    balA, balB = _side_balance(sigA), _side_balance(sigB)

    diff_exercises   = (exA != exB)
    diff_rules       = (rulesA != rulesB)
    diff_motifs      = (motA != motB)
    diff_blocks      = (blkA != blkB)
    diff_side_focus  = (focA != focB) or (balA != balB)

    diffs = {
        "diff_exercises": diff_exercises,
        "diff_rules": diff_rules,
        "diff_motifs": diff_motifs,
        "diff_blocks": diff_blocks,
        "diff_side_focus": diff_side_focus,
    }
    n_true = sum(1 for v in diffs.values() if v)
    diffs["overall_distinctness"] = 2 if n_true >= 2 else (1 if n_true == 1 else 0)
    return diffs

def bucket_pairs(items):
    # items: list of row indices/ids; return all unordered pairs
    return list(itertools.combinations(items, 2))

# ---------- main ----------
def main(input_files, use_llm_extraction=True, max_queries=4, samples_per_bucket_min=3,
         output_csv: Path | None = None, output_plot: Path | None = None,
         print_buckets: bool = False, cross_summary: bool = False):

    load_dotenv()

    # load dataframes and concat
    dfs = []
    for f in input_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
        if "generated_plan" not in df or "case_id" not in df:
            raise ValueError(f"{f} missing required fields.")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # derive grammar, size
    gs = df["case_id"].apply(parse_grammar_size)
    df["grammar"] = [g for g, s in gs]
    df["size"] = [s for g, s in gs]
    if "query_text" not in df:
        df["query_text"] = "<unknown-query>"

    # optional: downselect queries to first max_queries for speed
    kept_queries = list(df["query_text"].drop_duplicates())[:max_queries]
    df = df[df["query_text"].isin(kept_queries)].copy()

    # signature cache (persist across runs if you want)
    sig_cache = {}
    evaluator = LlmEvaluator(prompts_dir=PROJECT_ROOT / "evaluation" / "prompts") if use_llm_extraction else None

    def get_signature(plan_text: str):
        key = sha256(plan_text)
        if key in sig_cache:
            return sig_cache[key]
        # 1) try embedded
        sig = signature_from_embedded(plan_text)
        # 2) else LLM extraction if allowed
        if sig is None and use_llm_extraction:
            sig = evaluator.extract_signature(plan_text)
        # 3) fallback: empty signature
        if sig is None:
            sig = {"exercises": [], "rules": [], "motifs": [], "blocks": [], "focus": []}
        sig_cache[key] = sig
        return sig

    # attach signatures once per row
    df["signature"] = df["generated_plan"].apply(get_signature)

    # group by (grammar, size, query) -> compute DI
    records = []
    for (grammar, size, query), g in df.groupby(["grammar","size","query_text"]):
        # ensure enough samples
        if len(g) < samples_per_bucket_min:
            continue
        idxs = list(g.index)
        pairs = bucket_pairs(idxs)
        pair_diffs = []
        for i, j in pairs:
            sigA = df.at[i, "signature"]
            sigB = df.at[j, "signature"]
            diffs = compute_pair_diffs(sigA, sigB)
            pair_diffs.append(diffs)
        if not pair_diffs:
            continue
        # aggregate for this bucket
        facet_keys = ["diff_exercises","diff_rules","diff_motifs","diff_blocks","diff_side_focus"]
        means = {k: sum(1.0 if d[k] else 0.0 for d in pair_diffs)/len(pair_diffs) for k in facet_keys}
        di = sum(d["overall_distinctness"] for d in pair_diffs)/ (2.0*len(pair_diffs))  # scale 0..1
        records.append({
            "grammar": grammar,
            "size": size,
            "query": query,
            "diversity_index": di,
            **{f"mean_{k}": v for k, v in means.items()},
            "n_samples": len(g),
            "n_pairs": len(pair_diffs),
        })

    res = pd.DataFrame(records)
    if res.empty:
        print("No buckets met the minimum sample requirement. Generate >=3 samples per (grammar,size,query).")
        return

    if print_buckets:
        print("\nWITHIN-CORPUS buckets (each row is one (grammar,size,query) bucket):")
        print(res.sort_values(["size", "grammar", "query"]).to_string(index=False))

    # aggregate over queries -> one DI per (grammar,size)
    agg = res.groupby(["grammar","size"], as_index=False)["diversity_index"].mean()

    # sort sizes numerically
    agg = agg.sort_values(["grammar","size"])

    print("\nDiversity Index (mean over queries):")
    print(agg)

    # save CSV
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    # plot
    if output_plot:
        plt.figure()
        for grammar, g in agg.groupby("grammar"):
            plt.plot(g["size"], g["diversity_index"], marker="o", label=grammar)
        plt.xlabel("Corpus size")
        plt.ylabel("Diversity Index (0–1)")
        plt.title("Diversity vs Corpus Size (mean over queries)")
        plt.legend()
        Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot, bbox_inches="tight", dpi=160)
        print(f"Saved: {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of JSON files (different sizes) produced by your generator.")
    parser.add_argument("--no-llm", action="store_true",
                        help="If set, do NOT call LLM for signatures (expects embedded signatures).")
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Minimum samples per (grammar,size,query) to compute DI.")
    parser.add_argument("--print-buckets", action="store_true",
                        help="Print per-bucket within-corpus details (n_samples, n_pairs, facet means).")
    parser.add_argument("--cross-summary", action="store_true",
                        help="Also compute cross-corpus DI across grammars for the same (size,query).")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "experiments" / "diversity_vs_size.csv")
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "experiments" / "plots" / "diversity_vs_size.png")
    args = parser.parse_args()

    main(args.inputs, use_llm_extraction=not args.no_llm,
         max_queries=args.max_queries, samples_per_bucket_min=args.min_samples,
         output_csv=args.csv, output_plot=args.plot,
         print_buckets=args.print_buckets, cross_summary=args.cross_summary)
