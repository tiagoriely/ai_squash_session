# --- Hot-Patch for OpenMP/Threading Conflicts ---
# This section MUST be at the absolute top of the file
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("✅ Forcing single-threaded execution to prevent OpenMP conflicts.")
# --------------------------------------------------

import argparse, time, yaml, textwrap
from pathlib import Path
import torch
import torch.nn as nn

# --- monkey-patch cuda ---
if not torch.cuda.is_available():
    nn.Module.cuda    = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, **kw: self
# -------------------------

# Set torch threads after the environment variables
torch.set_num_threads(1)

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
    print(f"\n⏱️  {1000*(time.perf_counter()-t0):.1f} ms   |   top-k={len(docs)}\n")

    for r, (d, s) in enumerate(zip(docs, scores), 1):
        print(f"{r:2d}. id={d['id']:<4} score={s:6.4f}  source={(d['source'])}")