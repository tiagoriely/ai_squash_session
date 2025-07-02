# run: python3 -m pipelines.retrieval.run_field_retrieval

import re
import json
from pathlib import Path
from .field_matcher import parse_user_prompt, score_document # Import the logic


# Example Usage and Retrieval Logic (for demonstration purposes)
if __name__ == "__main__":
    KB_PATH = Path("data/my_kb.jsonl")

    if not KB_PATH.exists():
        print(f"Error: Knowledge base file not found at {KB_PATH}. Please run corpus_tools.py first.")
        exit()

    # Load your knowledge base
    knowledge_base = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            knowledge_base.append(json.loads(line))

    user_prompt = "I want an advanced drill for 2 players focusing on crosses and lobs with medium intensity lasting about 45 minutes."
    user_desires = parse_user_prompt(user_prompt)
    print(f"\nUser Desires: {user_desires}")

    scored_documents = []
    for doc in knowledge_base:
        s = score_document(doc, user_desires)
        scored_documents.append((s, doc))

    scored_documents.sort(key=lambda x: x[0], reverse=True)
    top_n_documents = [doc for score, doc in scored_documents if score > 0][:5]

    print("\n--- Top Relevant Documents (Field Retrieval) ---")
    if top_n_documents:
        for i, doc in enumerate(top_n_documents):
            score = next(s for s, d in scored_documents if d['id'] == doc['id'])
            print(f"Rank {i+1} (Score: {score:.2f}):")
            print(f"  ID: {doc.get('id')}, Source: {doc.get('source')}")
            print(f"  Type: {doc.get('type')}, Participants: {doc.get('participants')}, Level: {doc.get('squashLevel')}")
            print(f"  Intensity: {doc.get('intensity')}, Duration: {doc.get('duration')}")
            print(f"  Shots: {doc.get('shots')}, Shot Side: {doc.get('shotSide')}")
            print("-" * 20)
    else:
        print("No relevant documents found for the given prompt.")

    # Automated Evaluation of Generation (Conceptual, using top field-retrieved doc)
    # ... (rest of your automated evaluation code remains the same from previous eval_session_fields.py)
    if top_n_documents:
        generated_session_content = top_n_documents[0]['contents']
        generated_session_fields = top_n_documents[0]

        print(f"User Desires: {user_desires}")

        fulfilled_fields = 0
        total_requested_fields = 0

        fields_to_check = ['type', 'participants', 'squashLevel', 'intensity', 'duration', 'shots', 'shotSide']

        for field in fields_to_check:
            user_val = user_desires.get(field)
            generated_val = generated_session_fields.get(field)

            if user_val:
                total_requested_fields += 1
                if isinstance(user_val, list):
                    if set(user_val).issubset(set(generated_val or [])):
                        fulfilled_fields += 1
                elif field == 'duration':
                    try:
                        user_dur = int(re.search(r'\d+', user_val).group())
                        gen_dur = int(re.search(r'\d+', generated_val).group())
                        if abs(user_dur - gen_dur) <= 10:
                            fulfilled_fields += 1
                    except (ValueError, AttributeError):
                        pass
                elif generated_val == user_val:
                    fulfilled_fields += 1

        fulfillment_rate = (fulfilled_fields / total_requested_fields) * 100 if total_requested_fields > 0 else 0
        print(f"\nAutomated Field Fulfillment Rate: {fulfilled_fields}/{total_requested_fields} ({fulfillment_rate:.2f}%)")
    else:
        print("Cannot perform automated generation evaluation: No top documents found.")