# fore shared functions (e.g. loading data)
# evaluation/corpus_analysis/utils.py

import json
from pathlib import Path
from typing import List, Dict
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def load_corpus(filepath: Path) -> List[Dict]:
    """
    Loads a corpus from a .jsonl file, where each line is a JSON object.

    Args:
        filepath: The path to the .jsonl file.

    Returns:
        A list of dictionaries, where each dictionary represents a session.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Corpus file not found at: {filepath}")

    corpus = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode line in {filepath}: {line.strip()}")

    print(f"✅ Loaded {len(corpus)} sessions from {filepath.name}")
    return corpus

def count_total_variants(grammar_profile: str) -> int:
    """Counts the total number of variants defined in a grammar's exercise files."""
    grammar_path = Path(f"grammar/sports/squash/{grammar_profile}/exercises")
    if not grammar_path.is_dir():
        raise FileNotFoundError(f"Grammar exercise directory not found at: {grammar_path}")

    total_variants = 0
    for yaml_file in grammar_path.glob("*.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            doc = yaml.load(f)
            if doc and "variants" in doc:
                total_variants += len(doc["variants"])
    return total_variants

# Checking inside the Grammars
def load_exercise_library(grammar_profile: str) -> dict:
    """Loads all exercise variants from a grammar profile into a dictionary."""
    grammar_path = Path(f"grammar/sports/squash/{grammar_profile}/exercises")
    if not grammar_path.is_dir():
        raise FileNotFoundError(f"Grammar exercise directory not found at: {grammar_path}")

    library = {}
    for yaml_file in grammar_path.glob("*.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            doc = yaml.load(f)
            if doc and "variants" in doc:
                for variant in doc["variants"]:
                    if variant_id := variant.get("variant_id"):
                        library[variant_id] = variant
    print(f"📚 Loaded {len(library)} total variants from {grammar_profile} library.")
    return library