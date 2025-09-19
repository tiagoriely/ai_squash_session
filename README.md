# Repo Overview
```text
├── configs
├── data                      # Corpora in Processed, Generated Sessions Examples, Human-made sessions in Raw
├── evaluation                # Core Metrics in Utils (RAGAS, Self-BLEU, CG, ...) , Corpora Analysis (Clustering, Stats, ...)
├── experiments               # Experiment Results
├── field_adapters            # Metadata adapter for corpora
├── grammar
├── indexes                   # Sparse and Dense indexes for all Corpora
├── prompts
├── rag                       # RAG system core modules (excl. prompting, field retriever dictionary)
├── requirements.txt
├── scripts                   # Old scripts using rrf retriever
├── scripts2                  # Evaluations 
├── scripts_implementation    # implementation retrievers and prompting strategies
├── setup.py
├── src
├── tests                     # Testing PSA, Synthetic Corpora Generation Components (Grammar), Retrievers
├── third_party               # Used for Dense Retriever
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
• statistics: coverage, standard deviation, entropies, distributions, averages, adherence, ...
• DBSCAN: clustering visuals
```bash
$ python -m evaluation.corpus_analysis.run_all_analyses --config configs/corpus_analysis/statistics_analysis_config.yaml
$ python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering --config configs/corpus_analysis/clustering_config.yaml
```


# 3. Building Sparse and Dense Indexes

## Indexing Synthetic Corpus
```bash
$ python3 scripts/01_build_indexes.py \
    --semantic-build-config configs/indexing/build_semantic_index.yaml \
    --sparse-build-config configs/indexing/build_sparse_index_meta.yaml
```

## Indexing Manual Corpus
```bash
$ python3 scripts/01_build_indexes.py \
    --semantic-build-config configs/indexing/manual_semantic_index.yaml \
    --sparse-build-config configs/indexing/manual_sparse_index.yaml \
    --force
```


# 4. Generate Sessions with RAG

## Generating Session for Reliability and Structure Evaluation
top k = 10
temperature = 0.2
```bash
$ python scripts2/00_generate_outputs.py
```

## Generating Session for Diversity Evaluation
top k = 10
temperature = 0.7

```bash
$ python scripts2/02_generate_outputs_multi.py
```

# 5. Evaluate Diversity 

## SelfBleu + Disctinct-n

```bash
$ python3 scripts2/eval_selfbleu_distinctn_diversity.py
```

## Diversity Index
```bash
python3 scripts2/eval_diversity_vs_size.py \
  --input experiments/master_evaluation_file.json \
  --max-queries 4 \
  --min-samples 3
```

## Pairwise LLM Judge
```bash
$ python3 scripts2/eval_pairwise_diversity.py --input experiments/master_evaluation_file.json
```


# 6. Evaluate Reliability

## one file
python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/evaluation_sessions_set_k10_size500_*.json \
  --k 10 --by-query --csv experiments/ragas_scores_size500.csv

## many files (different sizes/grammars)
python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/dynamic_fusion_retrieval/*.json \
  --k 10 --by-query --csv experiments/ragas_scores_all.csv --plot-dir experiments/

## RAGAS with Raw Corpus
python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/evaluation_sessions_set_k10_rawCorpus_20250916_195637.json \
  --k 10 --by-query --csv experiments/ragas_scores_raw.csv --plot-dir experiments/

## Plot RAGAS with std
python scripts2/plot_ragas_with_std.py \
  --csv experiments/ragas/ragas_scores_all_1.csv \
  --out-csv experiments/ragas_agg_with_std_shade.csv \
  --out-dir experiments/plots

# 7. Evaluate Structure

## Completeness Gain for Structure
```Bash
$ python3 scripts2/eval_completeness_gain.py
```

## Programmatic Structure Adherence
```bash
$ python3 scripts2/eval_psa_structure.py
```

# Several Evaluations RunMarkDown
```bash
$ python3 scripts2/eval_pairwise_diversity.py --input experiments/master_evaluation_file.json
$ python scripts2/eval_reliability_with_ragas.py \
  --inputs experiments/dynamic_fusion_retrieval/*.json \
  --k 10 --by-query --csv experiments/ragas_scores_all.csv --plot-dir experiments/
$ python3 scripts2/eval_completeness_gain.py
$ python3 scripts2/eval_diversity_vs_size.py \
  --input experiments/master_evaluation_file.json \
  --max-queries 4 \
  --min-samples 3
$ python3 scripts2/eval_selfbleu_distinctn_diversity.py
```









