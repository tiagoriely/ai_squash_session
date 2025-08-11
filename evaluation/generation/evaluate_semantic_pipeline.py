# evaluation/generation/evaluate_semantic_pipeline.py

# --- Hot-Patch for OpenMP/Threading Conflicts ---
# This section MUST be at the absolute top of the file, before any other imports
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# --------------------------------------------------

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import torch

# --- RAGAS & Datasets Imports ---
from datasets import Dataset
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas import evaluate

# --- Environment ---
from dotenv import load_dotenv

load_dotenv()
torch.set_num_threads(1)

# --- Your RAG Components ---
from third_party.flashrag.flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI

# --- CONFIGURATION ---
QUERY_FILE_PATH = "data/squash_session_queries_prompts.json"
RETRIEVER_CONFIG_PATH = "rag/configs/retrieval/faiss_rerank.yaml"
PROMPT_TEMPLATE_PATH = "prompts/rag/session_conditioned_game.txt"
OUTPUT_DIR = "evaluation/evaluation_reports/RAGAS/semantic_only"
TOP_K_RETRIEVAL = 5


# --- HELPER FUNCTIONS ---
# (The rest of the script is unchanged)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_answer(prompt: str, model="gpt-4") -> str:
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


def run_full_rag_pipeline(queries: list, semantic_retriever, prompt_template: str) -> list:
    ragas_data = []
    total_queries = len(queries)

    for i, query in enumerate(queries, 1):
        print(f"\n--- 🔄 Processing query {i}/{total_queries}: \"{query[:80]}...\" ---")
        print("  - 🔎 Retrieving documents (Semantic Only)...")
        docs, _ = semantic_retriever.search(query, return_score=True)
        contexts = [doc['contents'] for doc in docs]

        if not contexts:
            print("  - ⚠️ No documents retrieved. Skipping generation.")
            contexts, answer = [], "No relevant documents were found."
        else:
            print(f"  - 🤖 Generating answer with {len(contexts)} contexts...")
            context_str = "\n---\n".join(contexts)
            generation_prompt = prompt_template.format(question=query, context=context_str)
            answer = generate_answer(generation_prompt)

        ragas_data.append({"question": query, "contexts": contexts, "answer": answer})

    return ragas_data


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full SEMANTIC-ONLY RAG pipeline evaluation.")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of prompts to evaluate. Set to -1 for all. (Default: 5)")
    args = parser.parse_args()

    print("🚀 Starting Semantic-Only RAG Pipeline Evaluation...")
    print("\n--- 📂 Loading assets ---")
    queries = load_json(QUERY_FILE_PATH)

    if args.limit != -1:
        print(f"✅ Limiting evaluation to the first {args.limit} prompts.")
        queries = queries[:args.limit]
    else:
        print(f"✅ Evaluating all {len(queries)} prompts.")

    retriever_config = load_yaml(RETRIEVER_CONFIG_PATH)
    retriever_config["retrieval_topk"] = TOP_K_RETRIEVAL
    print(f"✅ Set retriever top_k to: {TOP_K_RETRIEVAL}")

    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        prompt_template = f.read()

    print("--- 🧠 Initializing Semantic Retriever ---")
    semantic_retriever = DenseRetriever(retriever_config)

    start_time = time.time()
    evaluation_data_list = run_full_rag_pipeline(queries, semantic_retriever, prompt_template)
    end_time = time.time()
    print(f"\n--- ✅ Generation complete in {end_time - start_time:.2f} seconds ---")

    if not evaluation_data_list:
        print("🔴 No data was generated for evaluation. Exiting.")
        exit()

    print("\n--- 📊 Starting RAGAS evaluation ---")
    for item in evaluation_data_list:
        item['ground_truth'] = ""

    ragas_dataset = Dataset.from_list(evaluation_data_list)
    score_results = evaluate(ragas_dataset, metrics=[answer_relevancy, context_precision, faithfulness])
    print("--- ✅ RAGAS evaluation complete ---")

    print("\n--- 💾 Saving reports ---")
    eval_df = score_results.to_pandas()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline_name = "SemanticOnly"
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