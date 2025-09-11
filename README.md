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
$ pytest -v -s tests/
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
    --sparse-build-config configs/indexing/build_sparse_index_meta.yaml
```

python3 scripts/01_build_indexes.py \
    --semantic-build-config configs/indexing/build_semantic_index.yaml \
    --sparse-build-config configs/indexing/build_sparse_index_meta.yaml \
    --force

# 4. Generate Sessions with RAG

pick size and topk
```bash
$ python scripts2/00_generate_outputs.py
```

# 5. Evaluate Diversity 

## SelfBleu + Disctinct-n
pick json path to file based on size and topk
```bash
$ python scripts2/eval_combined_diversity_100.py
```

## Diversity Index
```bash
python scripts2/eval_diversity_vs_size.py \
  --inputs experiments/evaluation_sessions_set_k10_size100_samples3_20250907_213941.json \
          experiments/evaluation_sessions_set_k10_size300_samples3_20250907_220445.json \
          experiments/evaluation_sessions_set_k10_size500_samples3_20250907_234238.json \
  --max-queries 4 \
  --min-samples 3
```


# 6. Evaluate Reliability

python scripts2/eval_reliability_with_ragas.py \
  --input experiments/sample_query_1/evaluation_sessions_set_k10_size500_20250907_180221.json \
  --k 10 \
  --show-per-item \
  --csv experiments/ragas_scores_size500.csv

python scripts2/eval_reliability_with_ragas.py \
  --inputs evaluation_sessions_test.json \
  --k 10 \
  --by-query \
  --csv experiments/ragas_scores_all.csv \
  --plot-dir experiments/

## one file
python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/evaluation_sessions_set_k10_size500_*.json \
  --k 10 --by-query --csv experiments/ragas_scores_size500.csv

## many files (different sizes/grammars)
python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/sample_query_1/*.json \
  --k 10 --by-query --csv experiments/ragas_scores_all.csv --plot-dir experiments/

## Plot RAGAS with std
python scripts2/plot_ragas_with_std.py \
  --csv experiments/results/ragas_scores_all_1.csv \
  --out-csv experiments/ragas_agg_with_std_shade.csv \
  --out-dir experiments/plots




python scripts2/eval_diversity_vs_size.py --inputs experiments/evaluation_sessions_set_k10_size300_samples3_20250907_220445.json --min-samples 3 --max-queries 4 --print-buckets --cross-summary



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





