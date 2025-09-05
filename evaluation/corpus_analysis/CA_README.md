# Run all analysis

## Statistics
python -m evaluation.corpus_analysis.run_all_analyses --config configs/corpus_analysis/statistics_analysis_config.yaml

## DBSCAN
python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering --config configs/corpus_analysis/clustering_config.yaml




# Path to grammars
loose:      data/processed/loose_grammar/loose_{size}.jsonl \
balanced:   data/processed/balanced_grammar/balanced_{size}.jsonl \
strict:     data/processed/high_constraint_grammar/high_constraint_{size}.jsonl \
  high_constraint_grammar

# Statistics
## DIVERSITY
### Diversity 1 - loose 
python -m evaluation.corpus_analysis.statistics.measure_diversity \
  data/processed/loose_grammar/loose_500.jsonl \
  loose_grammar
### Diversity 2 - balanced
python -m evaluation.corpus_analysis.statistics.measure_diversity \
  data/processed/balanced_grammar/balanced_500.jsonl \
  balanced_grammar
### Diversity 3 -strict
python -m evaluation.corpus_analysis.statistics.measure_diversity \
  data/processed/high_constraint_grammar/high_constraint_500.jsonl \
  high_constraint_grammar

## Structure
### Structure 1 - loose
python -m evaluation.corpus_analysis.statistics.measure_structure data/processed/loose_grammar/loose_500.jsonl
### Structure 2 - balanced
python -m evaluation.corpus_analysis.statistics.measure_structure data/processed/balanced_grammar/balanced_500.jsonl
### Structure 3 - strict
python -m evaluation.corpus_analysis.statistics.measure_structure data/processed/high_constraint_grammar/high_constraint_500.jsonl

# Reliability
## Reliability 1 - loose
python -m evaluation.corpus_analysis.statistics.measure_reliability data/processed/loose_grammar/loose_500.jsonl loose_grammar

## Reliability 2 - balanced
python -m evaluation.corpus_analysis.statistics.measure_reliability data/processed/balanced_grammar/balanced_500.jsonl balanced_grammar

## Reliability 3 Strict
python -m evaluation.corpus_analysis.statistics.measure_reliability data/processed/high_constraint_grammar/high_constraint_500.jsonl high_constraint_grammar

# CLUSTERING
1. run Embedding
2. then run DBSCAN


## Loose Grammar DBSCAN
python -m evaluation.corpus_analysis.embeddings.embed_corpus data/processed/loose_grammar/loose_100.jsonl evaluation/corpus_analysis/embeddings/embed_size_100/loose_100.npy


python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_100/loose_100.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_100/loose_clusters_100.png \
  --eps 0.4 --min_samples 6

## Balanced Grammar DBSCAN
python -m evaluation.corpus_analysis.embeddings.embed_corpus data/processed/balanced_grammar/balanced_100.jsonl evaluation/corpus_analysis/embeddings/embed_size_100/balanced_100.npy

python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_100/balanced_100.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_100/balanced_clusters_100.png \
  --eps 0.4 --min_samples 6

## Strict Gammar DBSCAN
python -m evaluation.corpus_analysis.embeddings.embed_corpus data/processed/high_constraint_grammar/high_constraint_100.jsonl evaluation/corpus_analysis/embeddings/embed_size_100/high_constraint_100.npy  

python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_100/high_constraint_100.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_100/high_constraint_clusters_100.png \
  --eps 0.4 --min_samples 6




