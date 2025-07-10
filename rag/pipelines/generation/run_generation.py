# python rag/pipelines/generation/run_generation.py \
#        rag/configs/retrieval/faiss_rerank.yaml \
#        --query "I'm an intermediate player ..."


import argparse, time, yaml, textwrap, re
from pathlib import Path
from flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI
from collections import Counter
import os
import json # Ensure json is imported at the top if it wasn't already

# Environment
from dotenv import load_dotenv
load_dotenv()

# Patterns for session type inference
import re

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

    # ------------- NEW ► write / append ragas-ready row -------------
    # import json, pathlib # These are already at the top, no need to re-import
    out_path = Path("data/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # build the row
    ragas_row = {
        "question": args.query,
        "answer": answer,
        "contexts": [doc["contents"] for doc in docs],  # full texts shown to the LLM
    }

    current_data = [] # Initialize as an empty list

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8") as f: # Open in 'r' mode first
                current_data = json.load(f)
            # Ensure it's a list; if not, initialize as an empty list to prevent appending to non-list
            if not isinstance(current_data, list):
                print(f"⚠️ Warning: {out_path} content is not a JSON list. Reinitializing.")
                current_data = []
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding {out_path}: {e}")
            print(f"Attempting to reinitialize {out_path} as an empty list.")
            current_data = [] # If decoding fails, treat it as empty or corrupted
        except Exception as e:
            print(f"An unexpected error occurred when reading {out_path}: {e}")
            current_data = []

    # Append the new row
    current_data.append(ragas_row)

    # Write the complete (updated) data back to the file
    with out_path.open("w", encoding="utf-8") as f: # Open in 'w' mode to overwrite
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 1 row added to {out_path}")