# Create Sparse Retriever Index
python -m flashrag.retriever \ 
  --retrieval_method bm25 \
  --corpus_path data/processed/balanced_grammar/balanced_500.jsonl \
  --save_dir indexes/balanced_grammar/corpus_size_500

# GENERATOR
## run generator with retriever config
python3 -m rag.pipelines.generation.run_generation_with_random_prompt rag/configs/retrieval/faiss_rerank.yaml

# RETRIEVER
## run field retriever with corpus
 python3 -m rag.pipelines.retrieval.field_retrieval.run_field_retrieval --corpus squash_new 