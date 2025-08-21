# rag/pipelines/generation/run_generation.py
import argparse, time, yaml, textwrap, re
from pathlib import Path
from flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI
from collections import Counter
import os
import json

# Environment
from dotenv import load_dotenv
load_dotenv()

# Patterns for session type inference
SESSION_TYPES_PATTERNS = {
    "drill":            re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)|game\b", re.I),
    "solo":             re.compile(r"\bsolo\b", re.I),
    "ghosting":         re.compile(r"\bghosting\b", re.I),
    "mix":              re.compile(r"\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b.*\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b", re.I),
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


def generate_answer(prompt: str, model="gpt-4") -> str:
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
    print(
        "\n🤖 The session type isn't clear from your query.\n"
        "Please choose one of the following: drill, conditioned_game, solo, ghosting, mix"
    )
    answer = input("Your choice (or press Enter to use 'conditioned_game'): ").strip().lower()
    return answer if answer in SESSION_TYPES_PATTERNS else "conditioned_game"

# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for retriever")
    ap.add_argument("--query", required=True)
    ap.add_argument("--model", default="gpt-4")

    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    cfg.update({"use_reranker": True})

    retriever = DenseRetriever(cfg)

    t0 = time.perf_counter()
    docs, _ = retriever.search(args.query, return_score=True)
    t_elapsed = 1000 * (time.perf_counter() - t0)
    print(f"\n⏱  {t_elapsed:.1f} ms   |   top-k={len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i:2d}. id={doc['id']:<4} source={(doc['source'])}")

    print("\n=== Answer ===\n")
    context = build_context(docs)
    session_type = infer_type(args.query)

    prompt_path = Path("prompts/rag") / f"session_{session_type}.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    filled_prompt = prompt_template.format(question=args.query, context=context)
    answer = generate_answer(filled_prompt, model=args.model)
    print(answer)

    # ------------- write / append ragas-ready row -------------
    out_path = Path("data/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    current_data = []

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8") as f:
                current_data = json.load(f)
            if not isinstance(current_data, list):
                print(f"⚠️ Warning: {out_path} content is not a JSON list. Reinitializing.")
                current_data = []
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding {out_path}: {e}")
            print(f"Attempting to reinitialize {out_path} as an empty list.")
            current_data = []
        except Exception as e:
            print(f"An unexpected error occurred when reading {out_path}: {e}")
            current_data = []

    next_case_id = 1
    if current_data:
        max_existing_id = 0
        for row in current_data:
            if "case_id" in row and isinstance(row["case_id"], (int, str)):
                try:
                    max_existing_id = max(max_existing_id, int(row["case_id"]))
                except ValueError:
                    pass
        next_case_id = max_existing_id + 1

    # Prepare data for the RAGAS evaluation file
    # Store document metadata in a structured way
    retrieved_documents_info = [
        {"id": doc["id"], "source": doc["source"]} for doc in docs
    ]

    # build the row
    ragas_row = {
        "case_id": next_case_id,
        "question": args.query,
        "answer": answer,
        "contexts": [doc["contents"] for doc in docs],  # CRITICAL: This MUST be the full text.
        "retrieved_documents_info": retrieved_documents_info, # Consolidated, structured metadata.
        "reference": ""
    }

    # Append the new row
    current_data.append(ragas_row)

    # Write the complete (updated) data back to the file
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 1 row added to {out_path} with Case ID: {next_case_id}")