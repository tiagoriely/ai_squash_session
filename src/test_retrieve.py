from flashrag.retriever.retriever import DenseRetriever

cfg = {
    "retrieval_method":            "e5-base-v2",
    "retrieval_topk":              5,
    "index_path":                  "indexes/my_kb/e5-base-v2_Flat.index",
    "corpus_path":                 "data/my_kb.jsonl",
    "retrieval_model_path":        "intfloat/e5-base-v2",
    "retrieval_query_max_length":  64,
    "retrieval_pooling_method":    "mean",
    "retrieval_use_fp16":          False,
    "retrieval_batch_size":        32,
    "use_sentence_transformer":    False,
    "faiss_gpu":                   False,
    "use_reranker":                False,
    "save_retrieval_cache":        False,
    "use_retrieval_cache":         False,
    "instruction":                 None,
}
retriever = DenseRetriever(cfg)
for doc in retriever.search("cross-court lob drill"):
    print(f"[{doc['id']}] {doc['source']} → {doc['contents'][:120]}…")
