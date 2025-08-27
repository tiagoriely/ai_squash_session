
# 1. Build Indexes
```bash
$ python3 scripts/01_build_indexes.py \
    --semantic-build-config configs/indexing/build_semantic_index.yaml \
    --sparse-build-config configs/indexing/build_sparse_index.yaml
```





# 2. Run Experiment
## test semantic-only strategy

with gpt
```bash
$ python scripts/02_run_experiment.py \
    --query "A 60-minute solo practice for an intermediate player to improve drop shots." \
    --retrieval-strategy semantic_only \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --output-path data/results/semantic_only_results.jsonl
```

with ollama
```bash
$ python scripts/02_run_experiment.py \
    --query "A 60-minute session for an intermediate player to improve drop shots." \
    --retrieval-strategy semantic_only \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --output-path data/results/semantic_only_results.jsonl \
    --llm-model llama3
```

## test hybrid RRF strategy
```bash
$ python scripts/02_run_experiment.py \
    --query "A 60-minute solo practice for an intermediate player to improve drop shots." \
    --retrieval-strategy hybrid_rrf \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --field-config configs/retrieval/raw_squash_field_retrieval_config.yaml \
    --output-path data/results/rrf_results.jsonl
```
