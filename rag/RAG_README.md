
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
    --query "I need a 60-minute practice for an intermediate player to improve my counter drops." \
    --retrieval-strategy semantic_only \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --output-path data/results/semantic_only_results.jsonl
```

## test hybrid RRF strategy
```bash
$ python scripts/02_run_experiment.py \
    --query "I need a 60-minute practice for an intermediate player to improve my counter drops." \
    --retrieval-strategy hybrid_rrf \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --field-config configs/retrieval/raw_squash_field_retrieval_config.yaml \
    --output-path data/results/rrf_results.jsonl
```


python scripts/02_run_experiment.py \
    --query "I need a 60-minute practice for an intermediate player to improve my counter drops." \
    --retrieval-strategy sparse_only \
    --sparse-config configs/retrieval/sparse_retriever.yaml \
    --field-config configs/retrieval/raw_squash_field_retrieval_config.yaml \
    --output-path data/results/sparse_only_results.jsonl