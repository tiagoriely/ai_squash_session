# evaluation/retrieval/utils.py
import json
import os
from pathlib import Path

from field_adapters import SquashNewCorpusAdapter
from rag.retrieval.field_retriever import FieldRetriever
from rag.retrieval.semantic_retriever import SemanticRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.utils import load_and_format_config


def load_knowledge_base(corpus_path: str) -> list[dict]:
    """Loads the corpus from a JSONL file."""
    """Loads the corpus from a JSONL file and transforms it to the canonical format."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Knowledge base not found at: {corpus_path}")
    print(f"Loading and adapting knowledge base from: {corpus_path}")

    adapter = SquashNewCorpusAdapter()
    knowledge_base = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_doc = json.loads(line)
            # Transform each document as it's loaded
            transformed_doc = adapter.transform(raw_doc)
            knowledge_base.append(transformed_doc)

    print(f"   - Knowledge base loaded and adapted ({len(knowledge_base)} docs).")
    return knowledge_base


def load_all_query_sets(project_root: Path, grammar_type: str, corpus_size: int) -> list[dict]:
    """Finds and loads all generated query sets into a single list."""

    print("Loading all query sets...")
    all_queries = []
    query_set_dir = project_root / "evaluation" / "query_sets" / "generated"

    # 1. Load the static query sets
    static_sets = [
        "out_of_distribution.json",
        "under_specified.json",
        "graduated_complexity.json"
    ]
    for filename in static_sets:
        path = query_set_dir / filename
        with open(path, 'r') as f:
            all_queries.extend(json.load(f))
            print(f"   - Loaded {filename}")

    # 2. Load the dynamic "golden set" for the specific grammar
    golden_set_path = query_set_dir / grammar_type / str(corpus_size) / "golden_set.json"
    if golden_set_path.exists():
        with open(golden_set_path, 'r') as f:
            all_queries.extend(json.load(f))
            print(f"   - Loaded {golden_set_path.name} for {grammar_type}")
    else:
        print(f"   - WARNING: Golden set not found at {golden_set_path}")

    print(f"Total queries loaded: {len(all_queries)}")
    return all_queries


def initialise_retrievers(grammar_type: str, knowledge_base: list[dict], project_root: Path, corpus_size: int):
    """Initialises all retrievers for a specific grammar type using your actual config files."""
    print(f"Initialising retrievers for grammar: {grammar_type}...")

    template_context = {
        'grammar_type': grammar_type.replace('_grammar', ''),
        'corpus_size': corpus_size
    }

    # --- Use absolute paths for configs ---
    semantic_config_path = project_root / "configs" / "retrieval" / "semantic_retriever.yaml"
    sparse_config_path = project_root / "configs" / "retrieval" / "sparse_retriever.yaml"
    field_config_path = project_root / "configs" / "retrieval" / "raw_squash_field_retrieval_config.yaml"

    # --- Instantiate Retrievers ---
    semantic_config_raw = load_and_format_config(str(semantic_config_path), template_context)
    semantic_config_raw['corpus_path'] = str(project_root / semantic_config_raw['corpus_path'])
    semantic_config_raw['index_path'] = str(project_root / semantic_config_raw['index_path'])
    semantic_retriever = SemanticRetriever(config=semantic_config_raw)

    sparse_config_raw = load_and_format_config(str(sparse_config_path), template_context)
    sparse_config_raw['sparse_params']['index_path'] = str(
        project_root / sparse_config_raw['sparse_params']['index_path'])

    sparse_retriever = SparseRetriever(
        # This now correctly receives the already-adapted knowledge_base
        knowledge_base=knowledge_base,
        config=sparse_config_raw['sparse_params']
    )

    # The knowledge_base is already canonical, so we pass it directly
    field_retriever = FieldRetriever(
        knowledge_base=knowledge_base,
        config_path=str(field_config_path)
    )

    retrievers = {
        'semantic_e5': semantic_retriever,
        'sparse_bm25': sparse_retriever,
        'field_metadata': field_retriever
    }

    print("   - All retriever objects initialised successfully.")
    return retrievers
