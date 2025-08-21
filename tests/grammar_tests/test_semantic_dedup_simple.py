# test_semantic_dedup_simple.py
import sys
from pathlib import Path

# # Add the src directory to the Python path so we can import from it
# src_path = Path(__file__).parent / "src"
# sys.path.append(str(src_path))

try:
    from src.grammar_tools.dedup.semantic_dedup import SemanticDeduper, SemanticDedupNotAvailable
    print("✅ Successfully imported SemanticDeduper")
except SemanticDedupNotAvailable as e:
    print(f"❌ SemanticDedupNotAvailable on import: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"❌ Other Import Error: {e}")
    sys.exit(1)

# Initialize the deduper with a reasonable threshold
print("\n--- Testing Initialization ---")
try:
    deduper = SemanticDeduper(threshold=0.93) # Start with a sane value
    print("✅ SemanticDeduper initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

# Test with some simple, obviously similar texts
print("\n--- Testing with Sample Text ---")
sample_session_1 = "A 60-minute session focused on forehand drives and boasts. Includes warm-up and cool-down."
sample_session_2 = "A 60-minute session focused on forehand drives and boasts. Includes warm-up and cool-down." # Near duplicate
sample_session_3 = "A complete physical workout for cardio fitness, including running and weightlifting." # Obviously different

texts = [sample_session_1, sample_session_2, sample_session_3]

for i, text in enumerate(texts):
    print(f"\nText {i+1}: '{text}'")
    try:
        is_dup, max_sim, idx = deduper.is_near_duplicate(text)
        print(f"   is_near_duplicate: {is_dup}")
        print(f"   max_similarity: {max_sim:.4f}")
        print(f"   most_similar_index: {idx}")

        # Add the text to the deduper's history
        deduper.add(text)
        print(f"   -> Text added to deduper history.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")