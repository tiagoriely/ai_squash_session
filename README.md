# ai_squash_session

A lightweight playground that vendored **FlashRAG** so you can hack every line and run controlled experiments for **retrieval, reranking and generation** on your own documents.

---

## 📂 Repo anatomy

```text
ai_squash_session/
├── configs/
│   └── retrieval/              # YAML configs for each retrieval experiment
│       ├── faiss_base.yaml     # dense, no rerank
│       └── faiss_rerank.yaml   # dense + cross‑encoder rerank
│
├── data/
│   ├── raw/                    # drop .docx / .csv / .xlsx / .json here
│   └── my_kb.jsonl             # auto‑generated corpus
│
├── indexes/                    # FAISS (or other) index artefacts
│   └── my_kb/e5-base-v2_Flat.index
│
├── pipelines/
│   └── run_retrieval.py        # thin driver: YAML → DenseRetriever → stdout
│
├── scripts/
│   └── check_index.py          # optional FAISS inspector
│
├── src/
│   └── corpus_tools.py         # prepare_corpus() & helpers
│
├── tests/
│   └── test_retrieval.py       # parametrised over configs/retrieval/*.yaml
│
├── third_party/flashrag/       # vendored FlashRAG source, editable
├── requirements.txt
└── README.md   ← you are here
```

> **Why this layout?** Behaviour lives in YAML. Code paths stay generic. Add a new optimisation → just commit a new config and CI runs it automatically.

---

## 🚀 Quick‑start (retrieval‑only)

### 1 Clone & init submodule

```bash
# first time
$ git clone https://github.com/<you>/ai_squash_session.git && cd ai_squash_session
$ git submodule update --init --depth 1
```

### 2 Create venv & install deps

```bash
$ python -m venv venv && source venv/bin/activate
$ pip install --upgrade pip
# editable FlashRAG + runtime libs
$ pip install -e third_party/flashrag faiss-cpu datasets PyYAML transformers
```
*(Apple‑silicon: the code auto‑detects `mps`; Linux/NVIDIA: swap `faiss-cpu` for `faiss-gpu`, install CUDA PyTorch.)*

### 3 Add docs & build index

```bash
$ cp ~/Docs/*.docx data/raw/
$ python src/corpus_tools.py                # writes data/my_kb.jsonl
$ python -m flashrag.retriever.index_builder \
      --retrieval_method e5-base-v2 \
      --model_path intfloat/e5-base-v2 \
      --corpus_path data/my_kb.jsonl \
      --save_dir indexes/my_kb \
      --faiss_type Flat
```

### 4 Run baseline retrieval (config‑driven)

```bash
$ python pipelines/retrieval/run_semantic_retrieval.py \
       configs/retrieval/faiss_base.yaml \
       --query "cross-court lob drill"
```

### 5 Try the rerank variant

```bash
$ python pipelines/retrieval/run_semantic_retrieval.py configs/retrieval/faiss_rerank.yaml \
        --query "cross-court lob drill"
```

Output now shows rescored ordering.

### 6 Run the test‑suite & CI locally

```bash
$ pytest -q                 # runs every config under tests/
```

GitHub Actions (`.github/workflows/ci.yml`) repeats that on every push/PR—green badge means all configs still work.

---

## 🔮 Roadmap sections (placeholders you can fill later)

### Generation (`pipelines/run_generate.py`, `configs/generation/*`)
* retrieval → prompt building → LLM call
* supports vanilla GPT‑4, local Llama‑cpp, or FlashRAG’s built‑in generator.

```bash
$ python3 pipelines/generation/run_generation.py configs/retrieval/faiss_rerank.yaml \
  --query "Design a 60-minute squash session to improve my lobs"
```

### Hybrid BM25 + dense (coming)
* new YAML under `configs/retrieval/bm25_dense.yaml`
* extend `src/build_index.py` to build Pyserini text index.

### Evaluation metrics
* implement `src/eval.py` (nDCG@k, ROUGE‑L, BLEU) and call from tests or CI.

---

## ✏️ Maintaining your FlashRAG fork

| Task | Command |
|------|---------|
| **Edit code** | `vim third_party/flashrag/flashrag/retriever/utils.py` ➜ `git add -u && git commit` inside submodule ➜ `git add third_party/flashrag && git commit` in root |
| **Sync upstream** | `cd third_party/flashrag && git fetch origin && git merge origin/main` ➜ commit new submodule SHA |

Happy coding! 🚀
