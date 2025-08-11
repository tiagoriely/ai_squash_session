#run_generation_with_random_prompt.py
import argparse, time, yaml, textwrap, re
from pathlib import Path
from flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI
from collections import Counter
import os
import json
import random  # <-- Import the random module

# Environment
from dotenv import load_dotenv

load_dotenv()

# Patterns for session type inference
SESSION_TYPES_PATTERNS = {
    "drill": re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)|game\b", re.I),
    "solo": re.compile(r"\bsolo\b", re.I),
    "ghosting": re.compile(r"\bghosting\b", re.I),
    "mix": re.compile(
        r"\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b.*\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b", re.I),
}


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


def infer_type(query: str) -> str:
    for key, pattern in SESSION_TYPES_PATTERNS.items():
        if pattern.search(query):
            return key
    # If no type is inferred from the prompt, default to conditioned_game
    return "conditioned_game"


# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for retriever")
    # --- MODIFIED: --query is now optional ---
    ap.add_argument(
        "--query",
        required=False,
        default=None,
        help="The query to answer. If not provided, a random prompt is chosen from data/squash_session_queries_prompts.json"
    )
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    # --- NEW: Logic to select a query ---
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
    # --- MODIFIED: Use the 'query' variable ---
    docs, _ = retriever.search(query, return_score=True)
    t_elapsed = 1000 * (time.perf_counter() - t0)
    print(f"\n⏱  {t_elapsed:.1f} ms   |   top-k={len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i:2d}. id={doc['id']:<4} source={(doc['source'])}")

    print("\n=== Answer ===\n")
    context = build_context(docs)
    # --- MODIFIED: Use the 'query' variable ---
    session_type = infer_type(query)

    prompt_path = Path("prompts/rag") / f"session_{session_type}.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # --- MODIFIED: Use the 'query' variable ---
    filled_prompt = prompt_template.format(question=query, context=context)
    answer = generate_answer(filled_prompt, model=args.model)
    print(answer)

    # ... (The rest of the file for saving results remains the same) ...

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
    retrieved_documents_info = [{"id": doc["id"], "source": doc["source"]} for doc in docs]
    ragas_row = {
        "case_id": next_case_id,
        "question": query,  # --- MODIFIED: Use the 'query' variable ---
        "answer": answer,
        "contexts": [doc["contents"] for doc in docs],
        "retrieved_documents_info": retrieved_documents_info,
        "reference": ""
    }
    current_data.append(ragas_row)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 1 row added to {out_path} with Case ID: {next_case_id}")