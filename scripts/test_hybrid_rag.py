# scripts/test_hybrid_rag.py
"""
Simplified tester for the Hybrid RAG pipeline components.

This script allows you to test each retriever individually and the hybrid fusion,
see the parsed query fields, view the constructed prompt, and check the final answer.

Usage:
    python scripts/test_hybrid_rag.py --query "A 45-minute conditioned game" --corpus balanced_grammar --size 500
"""
import os
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict

# --- Environment settings for E5 model on CPU to avoid deadlocks ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Import your components ---
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval_fusion.strategies import query_aware_fusion
from rag.parsers.user_query_parser import parse_user_prompt
from rag.generation.prompt_constructor import DynamicPromptConstructor
from rag.generation.generator import Generator

# --- Load Env ---
from dotenv import load_dotenv

load_dotenv()


def load_knowledge_base(corpus_path: str) -> List[Dict]:
    """Load the knowledge base from a JSONL file."""
    knowledge_base = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                knowledge_base.append(json.loads(line))
    return knowledge_base


def initialize_retrievers(grammar_type: str, corpus_size: int, project_root: Path):
    """Initialize all retrievers with their configurations."""
    # Format the context for path templates
    grammar_type_short = grammar_type.replace('_grammar', '')
    context = {
        'grammar_type': grammar_type,
        'corpus_size': corpus_size,
        'grammar_type_short': grammar_type_short
    }

    # Load the knowledge base
    corpus_path_template = "data/processed/{grammar_type}/{grammar_type_short}_{corpus_size}.jsonl"
    corpus_path_str = corpus_path_template.format(**context)
    corpus_path = project_root / corpus_path_str

    print(f"Loading corpus from: {corpus_path}")
    knowledge_base = load_knowledge_base(corpus_path)
    print(f"Loaded knowledge base with {len(knowledge_base)} documents")

    # Initialize retrievers
    retrievers = {}

    # Field retriever
    field_config_path = project_root / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
    retrievers['field'] = FieldRetriever(knowledge_base, field_config_path)

    # Semantic retriever - need to format the config path
    semantic_config_path = project_root / "configs/retrieval/semantic_retriever.yaml"
    with open(semantic_config_path, 'r') as f:
        semantic_config = yaml.safe_load(f)

    # Override the corpus path in the semantic config to use our correct path
    semantic_config['corpus_path'] = str(corpus_path)

    # Create the correct index path based on your comment
    correct_index_path = f"indexes/{grammar_type}/corpus_size_{corpus_size}/e5/e5-base-v2_Flat_{grammar_type_short}.index"
    semantic_config['index_path'] = str(project_root / correct_index_path)

    print(f"Semantic config corpus path: {semantic_config['corpus_path']}")
    print(f"Semantic config index path: {semantic_config.get('index_path', 'Not set')}")

    try:
        retrievers['semantic'] = SemanticRetriever(semantic_config)
    except Exception as e:
        print(f"Failed to initialize semantic retriever: {e}")

    # Sparse retriever - need to format the index path
    sparse_config_path = project_root / "configs/retrieval/sparse_retriever.yaml"
    with open(sparse_config_path, 'r') as f:
        sparse_config = yaml.safe_load(f)

    # Create the correct index path based on your comment
    correct_sparse_index_path = f"indexes/{grammar_type}/corpus_size_{corpus_size}/bm25/bm25_index_{grammar_type_short}.pkl"
    sparse_config['index_path'] = str(project_root / correct_sparse_index_path)

    print(f"Sparse config index path: {sparse_config.get('index_path', 'Not set')}")

    try:
        retrievers['sparse'] = SparseRetriever(knowledge_base, sparse_config)
    except Exception as e:
        print(f"Failed to initialize sparse retriever: {e}")

    return retrievers, knowledge_base


def print_retrieval_results(retriever_name: str, results: List[Dict], top_k: int = 5):
    """Print retrieval results in a readable format."""
    print(f"\n{'=' * 80}")
    print(f"{retriever_name.upper()} RETRIEVER RESULTS (Top {top_k}):")
    print(f"{'=' * 80}")

    if not results:
        print("No results found")
        return

    for i, doc in enumerate(results[:top_k]):
        print(f"\n#{i + 1}:")
        print(f"  ID: {doc.get('id') or doc.get('session_id')}")
        print(f"  Title: {doc.get('title', 'N/A')}")

        # Print scores
        for key, value in doc.items():
            if '_score' in key:
                print(f"  {key}: {value:.4f}")

        # Print relevant metadata
        meta_keys = ['duration', 'participants', 'squashLevel', 'session_type', 'shots']
        for key in meta_keys:
            if key in doc:
                print(f"  {key}: {doc[key]}")


def main():
    parser = argparse.ArgumentParser(description="Test the Hybrid RAG pipeline components")
    parser.add_argument("--query", type=str, required=True, help="Query to test")
    parser.add_argument("--corpus", type=str, default="balanced_grammar",
                        choices=["loose_grammar", "balanced_grammar", "high_constraint_grammar"],
                        help="Corpus grammar type")
    parser.add_argument("--size", type=int, default=500, help="Corpus size")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # 1. Initialize retrievers
    print("Initializing retrievers...")
    try:
        retrievers, knowledge_base = initialize_retrievers(args.corpus, args.size, project_root)
    except Exception as e:
        print(f"Error initializing retrievers: {e}")
        print("Continuing with field retriever only...")
        # Try to at least initialize the field retriever
        context = {
            'grammar_type': args.corpus,
            'corpus_size': args.size,
            'grammar_type_short': args.corpus.replace('_grammar', '')
        }
        corpus_path_template = "data/processed/{grammar_type}/{grammar_type_short}_{corpus_size}.jsonl"
        corpus_path_str = corpus_path_template.format(**context)
        corpus_path = project_root / corpus_path_str
        knowledge_base = load_knowledge_base(corpus_path)

        field_config_path = project_root / "configs/retrieval/raw_squash_field_retrieval_config.yaml"
        retrievers = {'field': FieldRetriever(knowledge_base, field_config_path)}

    # 2. Parse the query
    print(f"\n{'=' * 80}")
    print("QUERY PARSING RESULTS:")
    print(f"{'=' * 80}")
    parsed_query = parse_user_prompt(args.query)
    print(f"Original query: {args.query}")
    print(f"Parsed fields: {json.dumps(parsed_query, indent=2)}")

    # 3. Test each retriever individually
    all_results = {}
    for retriever_name, retriever in retrievers.items():
        try:
            if retriever_name == 'field':
                # Field retriever needs allowed_durations parameter
                results = retriever.search(args.query, top_k=args.top_k, allowed_durations=None)
            else:
                results = retriever.search(args.query, top_k=args.top_k)

            all_results[retriever_name] = results
            print_retrieval_results(retriever_name, results, args.top_k)
        except Exception as e:
            print(f"Error with {retriever_name} retriever: {e}")

    # 4. Test hybrid fusion (if we have multiple retrievers)
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("HYBRID FUSION RESULTS:")
        print(f"{'=' * 80}")

        try:
            # Create a results map in the format expected by the fusion strategy
            ranked_lists_map = {}
            if 'semantic' in all_results:
                ranked_lists_map['semantic_e5'] = all_results.get('semantic', [])
            if 'sparse' in all_results:
                ranked_lists_map['sparse_bm25'] = all_results.get('sparse', [])
            if 'field' in all_results:
                ranked_lists_map['field_metadata'] = all_results.get('field', [])

            fused_results = query_aware_fusion(ranked_lists_map, args.query)
            print_retrieval_results("Hybrid Fusion", fused_results, args.top_k)
        except Exception as e:
            print(f"Error with hybrid fusion: {e}")
    else:
        print("Skipping hybrid fusion - only one retriever available")
        fused_results = list(all_results.values())[0] if all_results else []

    # 5. Test prompt construction
    print(f"\n{'=' * 80}")
    print("PROMPT CONSTRUCTION:")
    print(f"{'=' * 80}")

    try:
        prompt_constructor = DynamicPromptConstructor()
        prompt = prompt_constructor.create_prompt(args.query, fused_results[:args.top_k])
        print("Generated prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
    except Exception as e:
        print(f"Error with prompt construction: {e}")

    # 6. Test generation (optional - comment out if you don't want to call the LLM)
    print(f"\n{'=' * 80}")
    print("GENERATION (LLM RESPONSE):")
    print(f"{'=' * 80}")

    try:
        generator = Generator(model="gpt-4o")
        answer = generator.generate(prompt)
        print("Generated answer:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
    except Exception as e:
        print(f"Error with generation: {e}")


if __name__ == "__main__":
    main()