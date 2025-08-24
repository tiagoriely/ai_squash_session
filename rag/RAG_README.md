# Create Sparse Retriever Index
python -m flashrag.retriever \ 
  --retrieval_method bm25 \
  --corpus_path data/processed/balanced_grammar/balanced_500.jsonl \  #choose corpus
  --save_dir indexes/balanced_grammar/corpus_size_500  #choose index location

# Create Dense Retriever Index
$ export TOKENIZERS_PARALLELISM=false  #might be needed if running locally
$ export OMP_NUM_THREADS=1 #might be needed if running locally
$ python3 -m flashrag.retriever.index_builder \
    --retrieval_method e5-base-v2 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path data/processed/balanced_grammar/balanced_500.jsonl \
    --save_dir indexes/balanced_grammar/e5_index \
    --faiss_type Flat \
    --batch_size 8
$


# GENERATOR
## run generator with retriever config
python3 -m rag.pipelines.generation.run_generation_with_random_prompt rag/configs/retrieval/faiss_rerank.yaml

# RETRIEVER
## run field retriever with corpus
 python3 -m rag.pipelines.retrieval.field_retrieval.run_field_retrieval --corpus squash_new 