import json
import random
import argparse
from pathlib import Path

# Adjust the import path to find your utility function
from rag.utils import load_and_format_config


def generate_golden_queries(grammar_type: str):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    config_path = project_root / "configs" / "query_sets" / "golden_set_generator_config.yaml"

    # Load the config without formatting first to get the corpus_size
    base_config = load_and_format_config(str(config_path))
    corpus_size = base_config['corpus_size']

    # Create the context for formatting the paths in the config
    template_context = {
        'grammar_type': grammar_type.replace('_grammar', ''),
        'corpus_size': corpus_size  # This could also be read from the config if needed
    }

    # Load and format the config to get our paths and parameters
    config = load_and_format_config(str(config_path), template_context)

    corpus_path = project_root / config['corpus_path']
    output_path = project_root / config['query_set_path']
    num_samples = config['num_samples']

    # Load the knowledge base
    with open(corpus_path, "r", encoding="utf-8") as f:
        knowledge_base = [json.loads(line) for line in f]

    sampled_sessions = random.sample(knowledge_base, k=min(num_samples, len(knowledge_base)))

    queries = []
    for i, session in enumerate(sampled_sessions):
        duration = session.get("duration", "")
        level = session.get("squashLevel", "")
        participants = session.get("participants", "")
        primary_shots = ", ".join(session.get("primaryShots", []))

        prompt = f"A {duration}-minute session for {participants} {level} players, focusing on {primary_shots}."

        queries.append({
            "query_id": f"golden_{i + 1:02d}",
            "type": "golden_query",
            "text": prompt,
            "target_session_id": session.get("id") or session.get("session_id")
        })

    # Ensure the nested output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(queries, f, indent=2)

    print(f"✅ Generated {len(queries)} golden queries for '{grammar_type}' to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 'golden' queries from a corpus using a config file.")
    parser.add_argument("grammar_type", type=str, help="Name of the grammar (e.g., 'balanced_grammar').")
    args = parser.parse_args()
    generate_golden_queries(args.grammar_type)