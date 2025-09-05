# rag/retrieval_fusion/__init__.py

# This makes the fusion functions directly importable from the rag.fusion package
from .strategies import dynamic_query_aware_rrf, standard_unweighted_rrf, static_weighted_rrf