# ai\_squash\_session

A minimal sandbox that embeds **FlashRAG** (vendored in `third_party/flashrag`) so you can *own the code*, hack it, and use it to build a Retrieval‑Augmented Generation (RAG) stack on top of your private documents.

---

## 📂 Directory layout (after first run)

```text
ai_squash_session/
├── data/                  # raw + processed corpora
│   ├── raw/               #  ⤷ drop .docx / .csv / .xlsx / .json here
│   └── my_kb.jsonl        #  ⤷ generated JSONL corpus
├── indexes/               # FAISS / BM25 indexes land here
│   └── my_kb/
│       └── e5-base-v2_Flat.index
├── scripts/               # one‑off utilities
│   ├── prepare_corpus.py  # converts raw ⇒ JSONL
│   └── check_index.py     # tiny index inspector
├── src/                   # demo code you actually import/run
│   └── test_retrieve.py   # smoke test for retrieval
├── third_party/flashrag/  # full FlashRAG source, editable & versioned
└── README.md              # you are here
```

*(Everything else—virtual‑env, PyCharm settings, etc.—stays outside the repo.)*

---

## 🚀 Quick‑start: retrieval‑only pathway

### 1 Clone & pull FlashRAG as submodule

```bash
# one‑time setup
git clone https://github.com/<you>/ai_squash_session.git && cd ai_squash_session

git submodule update --init --depth 1   # pulls third_party/flashrag
```

### 2 Create a virtual‑env & install

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ./third_party/flashrag        # editable install of your FlashRAG copy
pip install faiss-cpu python-docx pandas openpyxl langid tqdm transformers
```

> **Apple‑silicon users:** no CUDA available; the code already detects `mps` (Metal) or falls back to CPU.
> **Linux/NVIDIA:** install `faiss-gpu` & the matching CUDA PyTorch wheel if you want GPU search.

### 3 Add files to `data/raw/`

Copy any `.docx`, `.csv`, `.xlsx`, or `.json` you care about into `data/raw/`.

### 4 Generate the corpus JSONL

```bash
python scripts/prepare_corpus.py                # writes data/my_kb.jsonl
```

### 5 Build a dense index (E5‑base)

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5-base-v2 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/my_kb.jsonl \
  --save_dir indexes/my_kb \
  --faiss_type Flat
```

*Patches already applied in `third_party/flashrag` ensure this runs on CPU/MPS.*

### 6 Smoke‑test retrieval

```bash
python src/test_retrieve.py
```

Expected sample output:

```text
[0] Drill - Lob cross (repetitive).docx → Duration: approx. 60‑70 min The session…
[1] practice_schedule.xlsx            → ### Sheet1 Time  Coach…
```

### 7 Inspect your index (optional)

```bash
python scripts/check_index.py
```

Shows vector count, dimension, and a quick nearest‑neighbour sanity check.

---

## 🔮 Next steps

### Generation (coming soon)

*Placeholder.*
You’ll plug `DenseRetriever` (or `MultiRetrieverRouter`) into an LLM prompt builder, maybe via the `flashrag.generator` pipeline.
Add your instructions here once you stabilise on a generator setup.

### Reranking

* Enable `use_reranker` in the config and add a cross‑encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Multi‑modal search

* Build two separate indexes with `--index_modal text` and `--index_modal image`; use `MultiModalRetriever`.

---

## ✏️ Maintaining your vendored FlashRAG

* **Edit** files in `third_party/flashrag/…`, commit inside the submodule, then:

  ```bash
  cd third_party/flashrag && git add -u && git commit -m "feat: …"
  cd ../.. && git add third_party/flashrag && git commit -m "Bump FlashRAG submodule"
  ```
* **Update**\* to latest upstream:

  ```bash
  cd third_party/flashrag
  git fetch origin && git merge origin/main
  cd ../..
  git add third_party/flashrag && git commit -m "FlashRAG upstream sync"
  ```

Happy hacking! 🚀
