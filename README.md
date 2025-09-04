# Repo Overview
```text
├── configs
├── data
├── evaluation
├── experiments
├── grammar
├── indexes
├── notes
├── prompts
├── rag
├── requirements.txt
├── scripts
├── setup.py
├── src
├── tests
├── third_party
└── venv

```



# Prerequisites
```bash
$ pip install -e .
$ pip install requirements.txt
```

# Useful to check errors in yaml files of grammars
$ yamllint .
$ ps aux | grep python
$ kill -9 <12345>



# 1. Create Corpuses
For this research three different corpuses are being generated, these are referred to as:
• loose grammar
• balanced grammar
• high constraint grammar

⚠️ in configs/synthetic_datasets_creation pick the size of the datasets with ```num```
```bash
$ python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/loose_grammar_run.yaml
$ python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/balanced_grammar_run.yaml
$ python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/high_constraint_grammar_run.yaml
```


# 2. Analyse Corpuses (Diversity, Reliability, Structure)

• statistics: coverage, standard deviation, entropies, distributions, averages, adherence
• DBSCAN: clustering visuals
```bash
$ python -m evaluation.corpus_analysis.run_all_analyses --config configs/corpus_analysis/statistics_analysis_config.yaml
$ python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering --config configs/corpus_analysis/clustering_config.yaml
```


# 3. Building Sparse and Dense Indexes
```bash
$ python3 scripts/01_build_indexes.py \
    --semantic-build-config configs/indexing/build_semantic_index.yaml \
    --sparse-build-config configs/indexing/build_sparse_index.yaml
```

# 4. Generate Sessions with RAG

## Sparse-only retriever + gpt-4o Generator
```bash
$ python scripts/03_run_experiment.py --config configs/retrieval/sparse_retriever.yaml
```

python scripts/06_run_full_experiment_hybrid_retriever.py --config configs/experiments/dissertation_experiment_config.yaml




# 5. Evaluate Results
python evaluation/evaluate_results.py \
  --input-file experiment/results/dissertation_run_sparse_only_v1_20250831_004009.jsonl \
  --output-file evaluation/results/test_evaluation.jsonl \
  --grammar-dir grammar/sports/squash/





