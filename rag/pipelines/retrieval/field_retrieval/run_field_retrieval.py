# run:  python3 -m rag.pipelines.retrieval.run_field_retrieval

import re
import json
from pathlib import Path
# Import the logic, and also the clean_and_standardise_value function for evaluation
from rag.pipelines.retrieval.field_retrieval.field_matcher import parse_user_prompt, score_document, clean_and_standardise_value


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

    user_prompt = "I want a conditioned game for 2 players focusing on cross lobs lasting about 45 minutes."
    print(f"User prompt: {user_prompt}")
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
            print(f"  Primary shots: {doc.get('primaryShots')}, Movement: {doc.get('movement')}")
            print("-" * 20)
    else:
        print("No relevant documents found for the given prompt.")

    # Automated Evaluation of Generation (Conceptual, using top field-retrieved doc)
    if top_n_documents:
        generated_session_content = top_n_documents[0]['contents'] # Using content of the top retrieved doc
        generated_session_fields = top_n_documents[0]             # Using fields of the top retrieved doc

        print(f"User Desires: {user_desires}")

        fulfilled_fields = 0
        total_requested_fields = 0

        # Define which fields to check for fulfilment. These should correspond to possible keys in user_desires.
        # 'primaryShots' and 'secondaryShots' are document characteristics, not direct user desires to be checked for fulfilment.
        fields_to_check = ['type', 'participants', 'squashLevel', 'intensity', 'duration', 'shots', 'shotSide', 'movement']
        # If parse_user_prompt starts populating 'explicitSpecificShots', you might add it here.
        # For now, 'shots' is assumed to contain both general and specific desired canonicals from the user.

        for field in fields_to_check:
            user_val = user_desires.get(field)
            raw_generated_val = generated_session_fields.get(field) # Get the raw value from the top document

            if user_val is not None: # Only check if the user actually requested something for this field
                total_requested_fields += 1

                # Standardise the generated document's value for a fair comparison.
                # This is crucial if the document stores "Medium" but user_desires has "medium", etc.
                # Use the field name (e.g., 'shots', 'duration') as the context for standardisation.
                standardised_generated_val = clean_and_standardise_value(field, raw_generated_val)


                if isinstance(user_val, list):
                    # For list fields (like 'shots', 'movement'), check if ALL user's desired items
                    # are a subset of the document's standardised list.
                    # Ensure generated_val_standardised is also a set for the subset check.
                    if set(user_val).issubset(set(standardised_generated_val or [])):
                        fulfilled_fields += 1
                elif field == 'duration':
                    # user_val (user_dur) is already an integer from parse_user_prompt
                    # standardised_generated_val (gen_dur) is also an integer from clean_and_standardise_value
                    user_dur = user_val
                    gen_dur = standardised_generated_val

                    if gen_dur is not None and abs(user_dur - gen_dur) <= 10: # Check if gen_dur is not None before comparison
                        fulfilled_fields += 1
                    # No need for try-except for conversion here, as clean_and_standardise_value handles it.
                else: # For exact match fields (like 'type', 'participants', 'squashLevel', 'intensity', 'shotSide')
                    if standardised_generated_val == user_val:
                        fulfilled_fields += 1

        fulfillment_rate = (fulfilled_fields / total_requested_fields) * 100 if total_requested_fields > 0 else 0
        print(f"\nAutomated Field Fulfilment Rate for best session: {fulfilled_fields}/{total_requested_fields} ({fulfillment_rate:.2f}%)")
    else:
        print("Cannot perform automated generation evaluation: No top documents found.")