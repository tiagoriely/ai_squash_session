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

        Args:
            query (str): The user's query.
            top_k (int): The number of documents to retrieve from each retriever.

        Returns:
            Dict: A dictionary containing the final answer and the retrieved documents.
        """
        # 1. --- RETRIEVAL ---
        # Get ranked lists from all retrievers
        all_ranked_lists = [r.search(query, top_k=top_k) for r in self.retrievers]

        # 2. --- FUSION ---
        # If there are multiple retrievers, fuse their results. Otherwise, use the single list.
        if len(all_ranked_lists) > 1:
            fused_docs = self.fusion_strategy(ranked_lists=all_ranked_lists)
        else:
            fused_docs = all_ranked_lists[0]

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