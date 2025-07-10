import faiss
from pathlib import Path
import datasets, textwrap

idx_path    = Path("indexes/my_kb/e5-base-v2_Flat.index")
corpus_path = Path("data/my_kb.jsonl")

# load index
index = faiss.read_index(str(idx_path))
print("----- FAISS index -----")
print("vectors stored :", index.ntotal)
print("dimension      :", index.d)
print("description    :", index)

# inspect first corpus row
corpus = datasets.load_dataset("json", data_files=str(corpus_path), split="train")
first  = corpus[0]
print("\n----- corpus[0] -----")
print("id       :", first["id"])
print("source   :", first.get("source"))
print("preview  :", textwrap.shorten(first["contents"].replace("\n", " "), 120))

# sanity-query the index
query_vec = index.reconstruct(0)         # use the first vector as its own query
scores, idxs = index.search(query_vec.reshape(1, -1), k=5)
print("\n----- nearest-neighbours to vector 0 -----")
for rank,(i,s) in enumerate(zip(idxs[0], scores[0]), 1):
    print(f"{rank:2d}. id={i:<5}  score={s:.4f}")
