# evaluation/generation/evaluate_hybrid_semantic_first_pipeline.py

import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# --- RAGAS & Datasets Imports ---
from datasets import Dataset
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas import evaluate

# --- Environment and Hot-Patches ---
from dotenv import load_dotenv

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch

torch.set_num_threads(1)

# --- Your RAG Components ---
from rag.pipelines.retrieval.hybrid_retrieval.hybrid_retriever_semantic_then_field import hybrid_search
from openai import OpenAI

# --- CONFIGURATION ---
QUERY_FILE_PATH = "data/squash_session_queries_prompts.json"
RETRIEVER_CONFIG_PATH = "rag/configs/retrieval/faiss_rerank.yaml"
PROMPT_TEMPLATE_PATH = "prompts/rag/session_conditioned_game.txt"
OUTPUT_DIR = "evaluation/evaluation_reports/RAGAS/hybrid"


# --- HELPER FUNCTIONS ---

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_answer(prompt: str, model="gpt-4") -> str:
    """Sends the prompt to the LLM and gets an answer."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful squash training assistant. You must ONLY use the information from the provided context documents to create the session plan."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  - 🔴 Error during generation: {e}")
        return "Error generating answer."


def run_full_rag_pipeline(
        queries: list,
        knowledge_base: list,
        semantic_retriever,
        prompt_template: str
) -> list:
    """
    Runs the full retrieve-then-generate process for a list of queries.
    """
    ragas_data = []
    total_queries = len(queries)

    for i, query in enumerate(queries, 1):
        print(f"\n--- 🔄 Processing query {i}/{total_queries}: \"{query[:80]}...\" ---")

        print("  - 🔎 Retrieving documents...")
        retrieved_results = hybrid_search(
            user_query=query,
            knowledge_base_docs=knowledge_base,
            semantic_retriever=semantic_retriever,
            semantic_threshold=0.6,
            final_top_k=5,
            alpha=0.7
        )
        contexts = [res['doc']['contents'] for res in retrieved_results]

        if not contexts:
            print("  - ⚠️ No documents retrieved. Skipping generation.")
            contexts = []
            answer = "No relevant documents were found to generate an answer."
        else:
            print(f"  - 🤖 Generating answer with {len(contexts)} contexts...")
            context_str = "\n---\n".join(contexts)
            generation_prompt = prompt_template.format(question=query, context=context_str)
            answer = generate_answer(generation_prompt)

        ragas_data.append({
            "question": query,
            "contexts": contexts,
            "answer": answer
        })

    return ragas_data


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # NEW: Add argparse to handle command-line arguments for limiting prompts
    parser = argparse.ArgumentParser(description="Run a full RAG pipeline evaluation using RAGAS.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of prompts to evaluate. Set to -1 to evaluate all prompts. (Default: 5)"
    )
    args = parser.parse_args()

    print("🚀 Starting Hybrid RAG Pipeline Evaluation...")

    print("\n--- 📂 Loading assets ---")
    queries = load_json(QUERY_FILE_PATH)

    # NEW: Limit the number of queries based on the --limit argument
    if args.limit != -1:
        print(f"✅ Limiting evaluation to the first {args.limit} prompts.")
        queries = queries[:args.limit]
    else:
        print(f"✅ Evaluating all {len(queries)} prompts.")

    retriever_config = load_yaml(RETRIEVER_CONFIG_PATH)
    kb_path = Path(retriever_config['corpus_path'])
    with open(kb_path, "r") as f:
        knowledge_base = [json.loads(line) for line in f if line.strip()]

    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        prompt_template = f.read()

    from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever

    print("--- 🧠 Initializing Semantic Retriever ---")
    semantic_retriever = DenseRetriever(retriever_config)

    start_time = time.time()
    evaluation_data_list = run_full_rag_pipeline(
        queries=queries,
        knowledge_base=knowledge_base,
        semantic_retriever=semantic_retriever,
        prompt_template=prompt_template
    )
    end_time = time.time()
    print(f"\n--- ✅ Generation complete in {end_time - start_time:.2f} seconds ---")

    if not evaluation_data_list:
        print("🔴 No data was generated for evaluation. Exiting.")
        exit()

    print("\n--- 📊 Starting RAGAS evaluation ---")
    for item in evaluation_data_list:
        item['ground_truth'] = ""

    ragas_dataset = Dataset.from_list(evaluation_data_list)

    score_results = evaluate(
        ragas_dataset,
        metrics=[answer_relevancy, context_precision, faithfulness],
    )
    print("--- ✅ RAGAS evaluation complete ---")

    print("\n--- 💾 Saving reports ---")
    eval_df = score_results.to_pandas()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = "SemanticFirst"

    # NEW: Add the number of evaluated prompts to the filename for clarity
    num_prompts_str = f"{len(queries)}prompts"

    excel_path = os.path.join(OUTPUT_DIR, f"ragas_summary_{pipeline_name}_{num_prompts_str}_{timestamp}.xlsx")
    json_path = os.path.join(OUTPUT_DIR, f"ragas_full_results_{pipeline_name}_{num_prompts_str}_{timestamp}.json")

    eval_df.to_excel(excel_path, index=False)
    print(f"  - Full summary report saved to: {excel_path}")

    eval_df.to_json(json_path, orient='records', indent=4)
    print(f"  - Full JSON results saved to: {json_path}")

    print("\n--- 📊 Final RAGAS Scores ---")
    print(score_results)
    print("\n🎉 Evaluation finished.")