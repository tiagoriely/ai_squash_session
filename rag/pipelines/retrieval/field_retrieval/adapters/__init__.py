# adapters/__init__.py
from .squash_old_corpus_adapter import SquashOldCorpusAdapter
from .squash_new_corpus_adapter import SquashNewCorpusAdapter

ADAPTER_MAP = {
    "squash_old": SquashOldCorpusAdapter,
    "squash_new": SquashNewCorpusAdapter,
    # Future: "tennis_v1": TennisV1Adapter,
}

def get_adapter(name: str):
    AdapterClass = ADAPTER_MAP.get(name)
    if not AdapterClass:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTER_MAP.keys())}")
    return AdapterClass()