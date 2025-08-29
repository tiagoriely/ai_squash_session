# scripts/03_run_experiment.py

"""
Main script for running RAG pipeline experiments.

Supports two modes of operation:
1. Config-driven: For self-contained experiments (e.g., sparse_only).
   `python scripts/02_run_experiment.py --config configs/retrieval/sparse_retriever.yaml`

2. Argument-driven: For other strategies (e.g., semantic_only).
   `python scripts/02_run_experiment.py --query "..." --retrieval-strategy semantic_only ...`
"""
import argparse
import json
import yaml
from pathlib import Path

# --- (Your RAG library imports remain the same) ---
from rag.pipeline import RAGPipeline
from rag.retrieval.semantic_retriever import SemanticRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval_fusion import reciprocal_rank_fusion
from rag.generation import PromptConstructor, Generator
from rag.utils import load_and_format_config  # We need this for the old method

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_knowledge_base(corpus_path: str) -> list[dict]:
    if not Path(corpus_path).exists():
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resolve_templates(config_node, variables):
    if isinstance(config_node, dict):
        return {k: resolve_templates(v, variables) for k, v in config_node.items()}
    elif isinstance(config_node, list):
        return [resolve_templates(item, variables) for item in config_node]
    elif isinstance(config_node, str):
        try:
            return config_node.format(**variables)
        except KeyError:
            return config_node
    else:
        return config_node


# --- Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a RAG pipeline experiment.")

    # --- Mode 1: New config-driven approach ---
    parser.add_argument("--config", help="Path to a standalone experiment YAML config file (for sparse_only).")

    # --- Mode 2: Original argument-driven approach ---
    parser.add_argument("--query", required=False, help="The user query to process.")
    parser.add_argument("--retrieval-strategy", required=False,
                        choices=['semantic_only', 'field_only', 'sparse_only', 'hybrid_rrf'],
                        help="The retrieval and fusion strategy to use.")
    parser.add_argument("--semantic-config", required=False, help="Path to the semantic retriever YAML config.")
    parser.add_argument("--sparse-config", required=False, help="Path to the sparse retriever YAML config.")
    parser.add_argument("--field-config", required=False, help="Path to the field retriever YAML config.")
    parser.add_argument("--corpus-path", required=False, help="Path to the JSONL corpus file.")
    parser.add_argument("--output-path", required=False, help="Path to save the output JSONL file.")

    # These arguments have defaults, so they are already optional.
    parser.add_argument("--prompt-template-dir", default="prompts/rag", help="Directory containing prompt templates.")
    parser.add_argument("--llm-model", default="gpt-4o", help="The language model to use for generation.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--rrf-k", type=int, default=60, help="The 'k' constant for Reciprocal Rank Fusion.")

    args = parser.parse_args()

    # --- Initialize variables ---
    params = {}

    if args.config:
        # --- MODE 1: Config File (for sparse_only) ---
        print(f"▶️  Running in Config Mode using: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        template_vars = config.get('template_vars', {})
        params = resolve_templates(config, template_vars)

        # We need a corpus path for the Sparse Retriever to load the actual documents
        if 'corpus' not in params.get('paths', {}):
            raise ValueError("Config file must contain a 'corpus' path under 'paths'.")
        params['corpus_path'] = params['paths']['corpus']
        params['output_path'] = params['paths']['output']
        params['prompt_template_dir'] = params['paths']['prompt_template_dir']

    else:
        # --- MODE 2: Command-Line Arguments (for semantic_only, etc.) ---
        print("▶️  Running in Command-Line Argument Mode")
        if not all([args.query, args.retrieval_strategy, args.corpus_path, args.output_path]):
            raise ValueError(
                "--query, --retrieval-strategy, --corpus-path, and --output-path are required in this mode.")
        # Map args to the same params dictionary for consistent use later
        params = vars(args)

    # --- Shared Pipeline Logic ---
    strategy = params['retrieval_strategy']
    knowledge_base = load_knowledge_base(params['corpus_path'])
    retrievers = []
    fusion_strategy = None

    print(f"   - Strategy: {strategy}")

    if strategy in ['semantic_only', 'hybrid_rrf']:
        print("   - Initializing SemanticRetriever...")
        if not params.get('semantic_config'): raise ValueError("--semantic-config is required for this strategy.")
        semantic_config_dict = load_and_format_config(params['semantic_config'])
        retrievers.append(SemanticRetriever(config=semantic_config_dict))

    if strategy in ['sparse_only', 'hybrid_rrf']:
        print("   - Initializing SparseRetriever...")

        if args.config:  # In config mode
            sparse_config_dict = params['sparse_params']
        else:  # In args mode
            if not params.get('sparse_config'): raise ValueError("--sparse-config is required for this strategy.")
            sparse_config_dict = load_and_format_config(params['sparse_config'])
        retrievers.append(SparseRetriever(knowledge_base=knowledge_base, config=sparse_config_dict))


    print("   - Initializing Generator and PromptConstructor...")
    generator = Generator(model=params['llm_model'])
    prompt_constructor = PromptConstructor(template_dir=params['prompt_template_dir'])

    pipeline = RAGPipeline(
        retrievers=retrievers,
        fusion_strategy=fusion_strategy,
        prompt_constructor=prompt_constructor,
        generator=generator
    )

    query = params['query']
    print(f"\n🚀 Running pipeline for query: '{query}'")
    result = pipeline.run(query=query, top_k=params['top_k'])

    output_path = Path(params['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ragas_entry = {"question": query, "answer": result["answer"],
                   "contexts": [doc.get("contents", "") for doc in result["retrieved_docs"]], "ground_truth": "",
                   "retrieved_documents_info": [{"id": doc.get("id")} for doc in result["retrieved_docs"]]}
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ragas_entry, ensure_ascii=False) + "\n")
    print(f"\n--- ✅ Experiment Complete --- \n   - Results saved to: {output_path}")