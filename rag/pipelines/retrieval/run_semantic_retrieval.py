import argparse, time, yaml, textwrap
from pathlib import Path

# --- monkey-patch ------------------------------------------------------------
""" this section is here to avoid using cuda"""
import torch
import torch.nn as nn

if not torch.cuda.is_available():        # keep real .cuda() on GPU runners
    nn.Module.cuda    = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self
# -----------------------------------------------------------------------------

from flashrag.retriever.retriever import DenseRetriever

def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def shorten(txt, w=100):
    return textwrap.shorten(txt.replace("\n", " "), w, placeholder="…")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="path to YAML (e.g. configs/retrieval/faiss_base.yaml)")
    ap.add_argument("--query", required=True)
    ap.add_argument("-k", "--topk", type=int)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.topk: cfg["retrieval_topk"] = args.topk

    retriever = DenseRetriever(cfg)

    t0 = time.perf_counter()
    docs, scores = retriever.search(args.query, return_score=True)
    print(f"\n⏱  {1000*(time.perf_counter()-t0):.1f} ms   |   top-k={len(docs)}\n")

    for r, (d, s) in enumerate(zip(docs, scores), 1):
        print(f"{r:2d}. id={d['id']:<4} score={s:6.4f}  source={(d['source'])}")
