# scripts/02_run_experiment.py
"""
Main script for running RAG pipeline experiments.

This script initializes the necessary components based on command-line arguments,
runs the RAG pipeline, and saves the results in a RAGAS-compatible format.

Example Usage:
python scripts/02_run_experiment.py \
    --query "I want an advanced drill for 2 players focusing on cross lobs." \
    --retrieval-strategy hybrid_rrf \
    --semantic-config configs/retrieval/semantic_retriever.yaml \
    --field-config configs/retrieval/field_retriever.yaml \
    --output-path data/results/experiment_results.jsonl
"""
import argparse
import json
import yaml
from pathlib import Path
import time

# --- Import from our new RAG library ---
from rag.pipeline import RAGPipeline
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.semantic_retriever import  SemanticRetriever
from rag.retrieval.sparse_retriever import  SparseRetriever
from rag.retrieval_fusion import (
    reciprocal_rank_fusion,
    rerank_by_weighted_score,
    sort_by_field_then_semantic
)
from rag.generation import PromptConstructor, Generator
from rag.utils import load_and_format_config

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_knowledge_base(corpus_path: str) -> list[dict]:
    if not Path(corpus_path).exists():
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG pipeline experiment.")
    parser.add_argument("--query", required=True, help="The user query to process.")
    parser.add_argument("--retrieval-strategy", required=True,
                        choices=['semantic_only', 'field_only', 'sparse_only', 'hybrid_rrf'],
                        help="The retrieval and fusion strategy to use.")

    # --- Configs and Paths ---
    parser.add_argument("--semantic-config", help="Path to the semantic retriever YAML config.")
    parser.add_argument("--sparse-config", help="Path to the sparse retriever YAML config.")
    parser.add_argument("--field-config", help="Path to the field retriever YAML config.")
    parser.add_argument("--corpus-path", required=True, help="Path to the JSONL corpus file.")
    parser.add_argument("--output-path", required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--prompt-template-dir", default="prompts/rag", help="Directory containing prompt templates.")

    # --- Model and Pipeline Parameters ---
    parser.add_argument("--llm-model", default="gpt-4o", help="The language model to use for generation.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--rrf-k", type=int, default=60, help="The 'k' constant for Reciprocal Rank Fusion.")
    args = parser.parse_args()

    print(f"▶️  Starting experiment with strategy: {args.retrieval_strategy}")

    # 1. --- Load Data and Configs ---
    print(f"   - Loading corpus from: {args.corpus_path}")
    knowledge_base = load_knowledge_base(args.corpus_path)

    # 2. --- Initialize Components based on Strategy ---
    retrievers = []
    fusion_strategy = None

    if args.retrieval_strategy in ['semantic_only', 'hybrid_rrf']:
        if not args.semantic_config: raise ValueError("--semantic-config is required for this strategy.")
        print("   - Initializing SemanticRetriever...")
        semantic_config = load_and_format_config(args.semantic_config)
        retrievers.append(SemanticRetriever(config=semantic_config))

    if args.retrieval_strategy in ['field_only', 'hybrid_rrf']:
        if not args.field_config: raise ValueError("--field-config is required for this strategy.")
        print("   - Initializing FieldRetriever...")
        retrievers.append(FieldRetriever(knowledge_base=knowledge_base, config_path=args.field_config))

    if args.retrieval_strategy in ['sparse_only', 'hybrid_rrf']:
        if not args.sparse_config: raise ValueError("--sparse-config is required for this strategy.")
        print("   - Initializing SparseRetriever...")
        sparse_config = load_and_format_config(args.sparse_config)
        retrievers.append(SparseRetriever(knowledge_base=knowledge_base, config=sparse_config))

    if args.retrieval_strategy == 'hybrid_rrf':
        fusion_strategy = lambda ranked_lists: reciprocal_rank_fusion(ranked_lists, k=args.rrf_k)

    print("   - Initializing Generator and PromptConstructor...")
    generator = Generator(model=args.llm_model)
    prompt_constructor = PromptConstructor(template_dir=args.prompt_template_dir)

    # 3. --- Initialize and Run the Pipeline ---
    pipeline = RAGPipeline(
        retrievers=retrievers,
        fusion_strategy=fusion_strategy,
        prompt_constructor=prompt_constructor,
        generator=generator
    )

    print(f"\n🚀 Running pipeline for query: '{args.query}'")
    result = pipeline.run(query=args.query, top_k=args.top_k)

    # 4. --- Save Results in RAGAS format ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ragas_entry = {
        "question": args.query,
        "answer": result["answer"],
        "contexts": [doc.get("contents", "") for doc in result["retrieved_docs"]],
        "ground_truth": "",
        "retrieved_documents_info": [{"id": doc.get("id")} for doc in result["retrieved_docs"]]
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ragas_entry, ensure_ascii=False) + "\n")

    print("\n--- ✅ Experiment Complete ---")
    if result["retrieved_docs"]:
        print(f"   - Top retrieved doc ID: {result['retrieved_docs'][0].get('id')}")
    else:
        print("   - ⚠️ No documents were retrieved.")
    print(f"   - Answer: {result['answer'][:100]}...")
    print(f"   - Results for 1 query saved to: {output_path}")


