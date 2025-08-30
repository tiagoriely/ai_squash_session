# rag/pipelines/generation/run_generation_with_random_prompt.py


import argparse, time, yaml, textwrap, re
from pathlib import Path
from flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI
from collections import Counter
import os
import json
import random

from rag_old.pipelines.retrieval.field_retrieval.session_type_inference import infer_session_type

# --- VERY IMPORTANT: this section is needed to use generator with local computer, if using CUDA comment the section
import torch
import os

# Force CPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent multiprocessing issues

# Monkey patch cuda() methods to use cpu instead
original_tensor_cuda = torch.Tensor.cuda
original_module_cuda = torch.nn.Module.cuda

def tensor_to_cpu(self, device=None, **kwargs):
    return self.to('cpu')

def module_to_cpu(self, device=None):
    return self.to('cpu')

torch.Tensor.cuda = tensor_to_cpu
torch.nn.Module.cuda = module_to_cpu

# Also override cuda availability check
torch.cuda.is_available = lambda: False
# ---------------------------------------------------------------------------------------------------

# Environment
from dotenv import load_dotenv

load_dotenv()


# Helpers
def load_cfg(path: str | Path) -> dict:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return data or {}


def shorten(txt: str, w: int = 100) -> str:
    return textwrap.shorten(txt.replace("\n", " "), w, placeholder="…")


def build_context(docs, max_tokens=3000):
    context_parts = []
    total = 0
    for doc in docs:
        tokens = len(doc["contents"].split())  # rough estimate: 1 word ≈ 1 token
        if total + tokens > max_tokens:
            break
        context_parts.append(doc["contents"])
        total += tokens
    return "\n---\n".join(context_parts)


def generate_answer(prompt: str, model="gpt-4o") -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful squash training assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# def infer_type(query: str) -> str:
#     for key, pattern in SESSION_TYPES_PATTERNS.items():
#         if pattern.search(query):
#             return key
#     # If no type is inferred from the prompt, default to conditioned_game
#     return "conditioned_game"


# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for retriever")
    ap.add_argument(
        "--query",
        required=False,
        default=None,
        help="The query to answer. If not provided, a random prompt is chosen from data/squash_session_queries_prompts.json"
    )
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    # Logic to select a query
    if args.query:
        query = args.query
        print(f"▶️ Using provided query: '{query}'")
    else:
        print("🤖 No query provided. Selecting a random prompt...")
        prompts_path = Path("data/squash_session_queries_prompts.json")
        if not prompts_path.exists():
            print(f"❌ Error: Prompts file not found at '{prompts_path}'")
            print("   Please run 'python evaluation/generation/create_prompts_for_generation.py' first.")
            exit()

        with prompts_path.open("r", encoding="utf-8") as f:
            prompts = json.load(f)

        if not prompts:
            print("❌ Error: The prompts file is empty.")
            exit()

        query = random.choice(prompts)
        print(f"   Randomly selected prompt: '{query}'")

    cfg = load_cfg(args.cfg)
    cfg.update({"use_reranker": True})

    retriever = DenseRetriever(cfg)

    t0 = time.perf_counter()
    docs, _ = retriever.search(query, return_score=True)
    t_elapsed = 1000 * (time.perf_counter() - t0)
    print(f"\n⏱  {t_elapsed:.1f} ms   |   top-k={len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i:2d}. id={doc['id']:<12}")

    print("\n=== Answer ===\n")
    context = build_context(docs)
    session_type = infer_session_type(query)

    prompt_path = Path("prompts/rag") / f"session_{session_type}.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    filled_prompt = prompt_template.format(question=query, context=context)
    answer = generate_answer(filled_prompt, model=args.model)
    print(answer)


    # ------------- write / append ragas-ready row -------------
    out_path = Path("data/processed/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    current_data = []
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8") as f:
                current_data = json.load(f)
            if not isinstance(current_data, list):
                current_data = []
        except (json.JSONDecodeError, Exception):
            current_data = []
    next_case_id = 1
    if current_data:
        max_existing_id = max(
            (int(row.get("case_id", 0)) for row in current_data if str(row.get("case_id", 0)).isdigit()), default=0)
        next_case_id = max_existing_id + 1
    retrieved_documents_info = [{"id": doc["id"]} for doc in docs]
    ragas_row = {
        "case_id": next_case_id,
        "question": query,
        "answer": answer,
        "contexts": [doc["contents"] for doc in docs],
        "retrieved_documents_info": retrieved_documents_info,
        "reference": ""
    }
    current_data.append(ragas_row)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 1 row added to {out_path} with Case ID: {next_case_id}")