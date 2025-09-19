import yaml
import json
from pathlib import Path

def generate_queries():
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / "out_of_distribution_keywords.yaml"
    output_path = base_path / "generated/out_of_distribution.json"

    with open(config_path, 'r') as f:
        keywords_config = yaml.safe_load(f)

    queries = []
    query_id_counter = 1
    for category, prompts in keywords_config.items():
        for prompt in prompts:
            queries.append({
                "query_id": f"ood_{query_id_counter:02d}", # ood = out-of-distribution
                "type": "out_of_distribution",
                "text": prompt
            })
            query_id_counter += 1

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(queries, f, indent=2)

    print(f"✅ Generated {len(queries)} out-of-distribution queries to {output_path}")

if __name__ == "__main__":
    generate_queries()