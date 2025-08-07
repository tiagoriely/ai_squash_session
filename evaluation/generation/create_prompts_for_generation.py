# evaluation/generation/create_prompts.py

import json
import random
from pathlib import Path
import re

# --- Configuration ---
KB_PATH = Path("data/my_kb.jsonl")
OUTPUT_PATH = Path("data/squash_session_queries_prompts.json")
NUM_PROMPTS_TO_GENERATE = 50


# --- Main Logic ---

def create_prompts(documents: list) -> set:
    """Generates a set of unique prompts based on document metadata."""
    generated_prompts = set()

    # Corrected templates to use the actual keys from your JSON (e.g., 'squashLevel', 'primaryShots')
    templates = [
        lambda
            meta: f"Design a {meta.get('duration', 'squash')} session to improve my {random.choice(meta['primaryShots'])}.",
        lambda
            meta: f"I need a {meta.get('squashLevel', 'new')} {meta.get('duration', 'squash')} session for {meta.get('participants', '2')} players.",
        lambda meta: f"Create a training routine that helps with '{meta['focus']}'.",
        lambda
            meta: f"Generate a {meta.get('intensity', 'medium')} intensity session for the {meta.get('shotSide', 'forehand')} that includes {random.choice(meta['primaryShots'])} and {random.choice(meta['secondaryShots'])}.",
        lambda
            meta: f"My opponent is strong on the attack. Give me a session to practice my defensive game, especially my {random.choice(meta.get('secondaryShots', ['lob']))}."
    ]

    max_attempts = len(documents) * 10
    attempts = 0
    while len(generated_prompts) < NUM_PROMPTS_TO_GENERATE and documents and attempts < max_attempts:
        # The 'doc' dictionary is our metadata. We just need to add the 'focus'.
        doc = random.choice(documents)
        attempts += 1

        # Safely extract the 'focus' from the contents field, as it's not a top-level key
        text_content = doc.get("contents", "")
        focus_match = re.search(r"Focus:(.*)", text_content)
        if focus_match:
            doc['focus'] = focus_match.group(1).strip()
        else:
            # If a doc has no Focus line, we can't use templates that require it.
            # We'll just skip this document for this attempt.
            continue

        try:
            template = random.choice(templates)
            # We pass the entire 'doc' object, which now contains the 'focus' key
            prompt = template(doc)
            generated_prompts.add(prompt)
        except (KeyError, IndexError, TypeError):
            # This will catch any documents that might be missing other required keys
            # (like 'primaryShots') for a specific template.
            continue

    return generated_prompts


def main():
    """Main function to load data, generate prompts, and save them."""
    print("🚀 Starting prompt generation...")
    if not KB_PATH.exists():
        print(f"❌ Error: Knowledge base file not found at '{KB_PATH}'")
        return

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            # Filter out any potentially empty lines
            documents = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {KB_PATH}: {e}")
        return

    print(f"✅ Loaded {len(documents)} documents from the knowledge base.")

    prompts = list(create_prompts(documents))

    if not prompts:
        print(
            "⚠️ Could not generate any prompts. Please check that documents in your KB have 'contents' with a 'Focus:' line and top-level metadata keys.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Successfully generated {len(prompts)} prompts!")
    print(f"   Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()