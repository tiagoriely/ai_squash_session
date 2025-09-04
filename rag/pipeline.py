# rag/pipeline.py

from typing import List, Dict, Callable
from .retrieval.base_retriever import BaseRetriever
from .generation.prompt_constructor import PromptConstructor
from .generation.generator import Generator


class RAGPipeline:
    """
    Orchestrates the entire Retrieve-Augment-Generate process.
    """

    def __init__(
            self,
            retrievers: List[BaseRetriever],
            fusion_strategy: Callable | None,
            prompt_constructor: PromptConstructor,
            generator: Generator
    ):
        """
        Initializes the RAG Pipeline with its components.

        Args:
            retrievers (List[BaseRetriever]): A list of initialized retriever objects.
            fusion_strategy (Callable | None): The function to use for fusing results.
                                               Can be None if only one retriever is used.
            prompt_constructor (PromptConstructor): An initialized prompt constructor object.
            generator (Generator): An initialized generator object for the LLM.
        """
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")
        if len(retrievers) > 1 and fusion_strategy is None:
            raise ValueError("A fusion_strategy must be provided when using multiple retrievers.")

        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.prompt_constructor = prompt_constructor
        self.generator = generator

    def run(self, query: str, top_k: int = 10) -> Dict:
        """
        Executes the full RAG pipeline for a given user query.
        """
        # 1. --- RETRIEVAL ---
        # Store results in a dictionary mapping retriever names to their docs
        all_ranked_lists_map = {
            # Using a simple name for the key
            r.__class__.__name__.replace("Retriever", "").lower(): r.search(query, top_k=top_k)
            for r in self.retrievers
        }

        # --- DEBUG START ---
        print("\n" + "=" * 50)
        print(f"DEBUG PIPELINE: Query -> '{query}'")
        print("--- Individual Retriever Results ---")
        for name, docs in all_ranked_lists_map.items():
            print(f"  - Retriever '{name}': Found {len(docs)} documents.")
        print("=" * 50 + "\n")

        # 2. --- FUSION ---
        if len(self.retrievers) > 1 and self.fusion_strategy:
            # Pass both the results map AND the original query to the fusion strategy
            fused_docs = self.fusion_strategy(ranked_lists_map=all_ranked_lists_map, query=query)
        else:
            # Get the single list from the dictionary
            fused_docs = next(iter(all_ranked_lists_map.values()))

        # Ensure we only pass the top_k fused documents to the context
        final_docs = fused_docs[:top_k]

        # 3. --- PROMPT CONSTRUCTION ---
        # Create the final prompt using the query and the fused documents
        final_prompt = self.prompt_constructor.create_prompt(query, final_docs)

        # 4. --- GENERATION ---
        # Get the final answer from the LLM
        answer = self.generator.generate(final_prompt)

        return {
            "answer": answer,
            "retrieved_docs": final_docs
        }