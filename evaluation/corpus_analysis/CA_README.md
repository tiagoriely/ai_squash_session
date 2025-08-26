# Path to grammars
loose:      data/processed/loose_grammar/loose_500.jsonl \
balanced:   data/processed/balanced_grammar/balanced_500.jsonl \
strict:     data/processed/high_constraint_grammar/high_constraint_500.jsonl \
  high_constraint_grammar

# Statistics

## Diversity

1. Lexical Diversity
### Grammar 1 - loose
python -m evaluation.corpus_analysis.diversity.lexical_metrics \
  data/processed/loose_grammar/loose_500.jsonl \
  loose_grammar
### Grammar 2 - balanced
python -m evaluation.corpus_analysis.diversity.lexical_metrics \
  data/processed/balanced_grammar/balanced_500.jsonl \
  balanced_grammar
### Grammar 3 -strict
python -m evaluation.corpus_analysis.diversity.lexical_metrics \
  data/processed/high_constraint_grammar/high_constraint_500.jsonl \
  high_constraint_grammar

# Clustering
1. run Embedding
2. then run DBSCAN


## Loose Grammar DBSCAN
python -m evaluation.corpus_analysis.embedding.embed_corpus data/processed/loose_grammar/loose_500.jsonl evaluation/corpus_analysis/embeddings/embed_size_500/loose_500.npy


python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_500/loose_500.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_500/loose_clusters_500.png \
  --eps 0.4 --min_samples 6

## Balanced Grammar DBSCAN
python -m evaluation.corpus_analysis.embedding.embed_corpus data/processed/balanced_grammar/balanced_500.jsonl evaluation/corpus_analysis/embeddings/embed_size_500/balanced_500.npy

python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_500/balanced_500.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_500/balanced_clusters_500.png \
  --eps 0.4 --min_samples 6

## Strict Gammar DBSCAN
python -m evaluation.corpus_analysis.embedding.embed_corpus \             
  data/processed/high_constraint_grammar/high_constraint_500.jsonl \            
  evaluation/corpus_analysis/embeddings/embed_size_500/high_constraint_500.npy  

python -m evaluation.corpus_analysis.clustering.run_dbscan_clustering \
  evaluation/corpus_analysis/embeddings/embed_size_500/high_constraint_500.npy \
  evaluation/corpus_analysis/visualisations/corpus_size_500/high_constraint_clusters_500.png \
  --eps 0.4 --min_samples 6




