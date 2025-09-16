import pandas as pd
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# Import retrieval strategy
from rag.retrieval_fusion.strategies import query_aware_fusion  # Our new fusion strategy
from rag.utils import load_and_format_config

# Import helper functions from your main evaluation script
from .utils import load_knowledge_base, load_all_query_sets, initialise_retrievers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_hybrid_evaluation(grammar_type: str):
    """Runs the evaluation for the complete hybrid retrieval system."""

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    base_config = load_and_format_config(str(config_path))
    corpus_size = base_config['corpus_size']
    template_context = {'grammar_type': grammar_type.replace('_grammar', ''), 'corpus_size': corpus_size}
    formatted_config = load_and_format_config(str(config_path), template_context)
    corpus_path = project_root / formatted_config['corpus_path']

    # 1. Load data and queries
    knowledge_base = load_knowledge_base(str(corpus_path))
    queries = load_all_query_sets(project_root, grammar_type, corpus_size)

    # 2. Initialize all three retrievers
    retrievers_map = initialise_retrievers(grammar_type, knowledge_base, project_root, corpus_size)

    all_results = []
    print(f"\nRunning HYBRID evaluation for {len(queries)} total queries...")

    for query_info in tqdm(queries, desc="Processing Queries with Hybrid Retriever"):
        query_text = query_info['text']

        # 3. Retrieve from all sources
        retrieved_lists_map = {
            name: retriever.search(query=query_text, top_k=30)  # Retrieve more candidates for fusion
            for name, retriever in retrievers_map.items()
        }

        # 4. Fuse the results using our new strategy
        fused_docs = query_aware_fusion(ranked_lists_map=retrieved_lists_map, query=query_text)

        # 5. Store the final fused results
        for rank, doc in enumerate(fused_docs[:10]):  # Store top 10 fused results
            all_results.append({
                'query_id': query_info['query_id'],
                'query_type': query_info['type'],
                'query_text': query_text,
                'retriever_name': f"hybrid_fusion_{grammar_type}",
                'rank': rank + 1,
                'document_id': doc.get('id') or doc.get('session_id'),
                'score': doc.get('fusion_score')  # Use the new fusion_score
            })

    # 6. Save the final CSV
    results_df = pd.DataFrame(all_results)
    output_dir = project_root / "evaluation" / "retrieval" / grammar_type / f"corpus_size_{corpus_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"retrieval_results_hybrid-fusion_{grammar_type}_{corpus_size}.csv"

    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Hybrid evaluation complete! Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid retriever evaluation.")
    parser.add_argument("grammar", type=str, help="The grammar to evaluate.")
    args = parser.parse_args()
    run_hybrid_evaluation(grammar_type=args.grammar)