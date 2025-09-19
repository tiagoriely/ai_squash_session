# rag/pipeline.py (Updated Version)

from typing import List, Dict, Callable
from .retrieval.base_retriever import BaseRetriever
from .generation.prompt_constructor import PromptConstructor_v2
from .generation.generator import Generator


class RAGPipeline:
    """
    Orchestrates the entire Retrieve-Augment-Generate process.
    """

    def __init__(
            self,
            retrievers: List[BaseRetriever],
            fusion_strategy: Callable | None,
            prompt_constructor: PromptConstructor_v2,
            generator: Generator
    ):
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")
        if len(retrievers) > 1 and fusion_strategy is None:
            raise ValueError("A fusion_strategy must be provided when using multiple retrievers.")

        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.prompt_constructor = prompt_constructor
        self.generator = generator

    def run(self, query: str, retrieval_k: int = 30, context_k: int = 3) -> Dict:
        """
        Executes the full RAG pipeline for a given user query.
        """
        # 1. --- RETRIEVAL ---
        all_ranked_lists_map = {
            r.__class__.__name__.replace("Retriever", "").lower(): r.search(query, top_k=retrieval_k)
            for r in self.retrievers
        }

        # --- DEBUG ---
        print(f"\nDEBUG: Retrieved docs from {len(all_ranked_lists_map)} sources with retrieval_k={retrieval_k}")

        # 2. --- FUSION ---
        if len(self.retrievers) > 1 and self.fusion_strategy:
            fused_docs = self.fusion_strategy(ranked_lists=all_ranked_lists_map, query=query)
        else:
            fused_docs = next(iter(all_ranked_lists_map.values()))

        # 3. --- CONTEXT SELECTION ---
        final_docs = fused_docs[:context_k]
        print(f"DEBUG: Selected final {len(final_docs)} documents for context with context_k={context_k}")

        # 4. --- PROMPT CONSTRUCTION ---
        final_prompt = self.prompt_constructor.create_prompt(query, final_docs)

        # 5. --- GENERATION ---
        answer = self.generator.generate(final_prompt)

        return {
            "answer": answer,
            "retrieved_docs": final_docs
        }