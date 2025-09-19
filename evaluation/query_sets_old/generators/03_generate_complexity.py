import json
from pathlib import Path


def generate_queries():
    base_path = Path(__file__).resolve().parent.parent

    # --- Strategy 3: Under-specified ---
    under_specified_queries = [
        {"query_id": "under_spec_01", "type": "under_specified", "text": "generate a squash session"},
        {"query_id": "under_spec_02", "type": "under_specified", "text": "a drill for squash"}
    ]
    output_path_us = base_path / "generated/under_specified.json"
    with open(output_path_us, 'w') as f:
        json.dump(under_specified_queries, f, indent=2)
    print(f"✅ Generated {len(under_specified_queries)} under-specified queries to {output_path_us}")

    # --- Strategy 4: Graduated Complexity ---
    graduated_queries = [
        {"query_id": "complex_01", "type": "graduated_complexity", "text": "a conditioned game session"},
        {"query_id": "complex_02", "type": "graduated_complexity", "text": "a 45-minute conditioned game session"},
        {"query_id": "complex_03", "type": "graduated_complexity", "text": "a 45-minute conditioned game session for an advanced player"},
        {"query_id": "complex_04", "type": "graduated_complexity", "text": "a 45-minute conditioned game session for an advanced player focusing on volley drops"}
    ]
    output_path_gc = base_path / "generated/graduated_complexity.json"
    with open(output_path_gc, 'w') as f:
        json.dump(graduated_queries, f, indent=2)
    print(f"✅ Generated {len(graduated_queries)} graduated complexity queries to {output_path_gc}")


if __name__ == "__main__":
    generate_queries()