# src/grammar_tools/loader.py
from pathlib import Path
from ruamel.yaml import YAML
from pydantic import BaseModel, Field

yaml = YAML(typ="safe")

class Activity(BaseModel):
    id: str = Field(..., alias="activity_id")
    name: str
    is_abstract: bool = False
    extends: list[str] | None = None
    defaults: dict | None = None
    allowed_actions: list[str] | None = None
    rules: dict | None = None

CATALOG_ROOT = Path("data/grammar/sports/squash")

def _iter_yaml_files():
    yield from CATALOG_ROOT.rglob("*.yaml")

def load_yaml_dir() -> dict[str, Activity]:
    acts: dict[str, Activity] = {}
    for file in _iter_yaml_files():
        doc = yaml.load(file.read_text()) or {}
        # pick the first recognised top-level key
        key = next((k for k in ("activities", "content") if k in doc), None)
        if not key:
            # templates / defaults – skip for session generation
            continue
        for node in doc[key]:
            if "activity_id" not in node:          # e.g. template node
                continue
            act = Activity.model_validate(node)
            acts[act.id] = act
    return acts
