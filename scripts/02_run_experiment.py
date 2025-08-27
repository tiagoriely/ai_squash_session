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
from rag.retrieval_fusion import (
    reciprocal_rank_fusion,
    rerank_by_weighted_score,
    sort_by_field_then_semantic
)
from rag.generation import PromptConstructor, Generator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- Helper Functions ---

def load_and_format_config(config_path: str) -> dict:
    """
    Loads a YAML config file and formats any string values that contain
    placeholders (e.g., {variable_name}) using other keys from the config.
    This version handles nested structures.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Helper to recursively format the values
    def _format_values(obj, context):
        if isinstance(obj, dict):
            return {k: _format_values(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_format_values(elem, context) for elem in obj]
        elif isinstance(obj, str):
            try:
                return obj.format(**context)
            except KeyError:
                # This string doesn't have a placeholder, so return as is
                return obj
        else:
            return obj

    # Start the recursive formatting
    return _format_values(config, config)


def load_knowledge_base(corpus_path: str) -> list[dict]:
    """Loads the knowledge base from a JSONL file."""
    if not Path(corpus_path).exists():
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG pipeline experiment.")
    parser.add_argument("--query", required=True, help="The user query to process.")
    parser.add_argument("--retrieval-strategy", required=True,
                        choices=['semantic_only', 'field_only', 'hybrid_rrf', 'hybrid_weighted_rerank',
                                 'hybrid_cascade'],
                        help="The retrieval and fusion strategy to use.")

    # --- Configs and Paths ---
    parser.add_argument("--semantic-config", help="Path to the semantic retriever YAML config.")
    parser.add_argument("--field-config", help="Path to the field retriever YAML config.")
    parser.add_argument("--output-path", required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--prompt-template-dir", default="prompts/rag", help="Directory containing prompt templates.")

    # --- Model and Pipeline Parameters ---
    parser.add_argument("--llm-model", default="gpt-4o", help="The language model to use for generation.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for the field score in weighted reranking.")
    parser.add_argument("--rrf-k", type=int, default=60, help="The 'k' constant for Reciprocal Rank Fusion.")

    args = parser.parse_args()

    print(f"▶️  Starting experiment with strategy: {args.retrieval_strategy}")
    t0_total = time.perf_counter()

    # 1. --- Load Data and Configs ---
    semantic_config = load_and_format_config(args.semantic_config) if args.semantic_config else {}
    field_config_path = args.field_config or "configs/retrieval/raw_squash_field_retrieval_config.yaml"

    # The knowledge base path is taken from the semantic config as the primary source of truth
    kb_path = semantic_config.get('corpus_path')
    if not kb_path:
        raise ValueError("`corpus_path` must be defined in the semantic config.")
    knowledge_base = load_knowledge_base(kb_path)

    # 2. --- Initialize Components based on Strategy ---
    retrievers = []
    fusion_strategy = None

    if args.retrieval_strategy in ['semantic_only', 'hybrid_rrf', 'hybrid_weighted_rerank', 'hybrid_cascade']:
        print("   - Initializing SemanticRetriever...")
        retrievers.append(SemanticRetriever(config=semantic_config))

    if args.retrieval_strategy in ['field_only', 'hybrid_rrf', 'hybrid_weighted_rerank', 'hybrid_cascade']:
        print("   - Initializing FieldRetriever...")
        retrievers.append(FieldRetriever(knowledge_base=knowledge_base, config_path=field_config_path))

    if args.retrieval_strategy == 'hybrid_rrf':
        fusion_strategy = lambda ranked_lists: reciprocal_rank_fusion(ranked_lists, k=args.rrf_k)
    elif args.retrieval_strategy == 'hybrid_weighted_rerank':
        # This strategy requires a custom pipeline step, so we'll handle it inside the pipeline later
        # For now, we indicate no fusion, as the re-ranking happens on a combined list
        pass
    elif args.retrieval_strategy == 'hybrid_cascade':
        pass  # Similar custom logic needed

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
        "contexts": [doc["contents"] for doc in result["retrieved_docs"]],
        "ground_truth": "",  # Placeholder for golden answer
        "retrieved_documents_info": [{"id": doc["id"]} for doc in result["retrieved_docs"]]
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ragas_entry, ensure_ascii=False) + "\n")

    t1_total = time.perf_counter()
    print("\n--- ✅ Experiment Complete ---")
    print(f"   - Answer: {result['answer'][:100]}...")
    print(f"   - Results for 1 query saved to: {output_path}")
    print(f"   - Total time: {t1_total - t0_total:.2f} seconds")