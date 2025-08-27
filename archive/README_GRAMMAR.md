# strict matching ("extra drive" must be listed exactly)
python -m src.grammar_tools.loader --policy exact --list-errors --summary

# base-family matching allowed (e.g., "drive" covers "extra drive")
python -m src.grammar_tools.loader --policy family --list-errors --summary

# default hybrid policy (exact OR family)
python -m src.grammar_tools.loader --list-errors --summary

# Generate a Synthetic Dataset using the Grammar
python -m src.grammar_tools.grammar_generator -n 100 -m 60 -o data/processed/grammar_synthetic_dataset.jsonl

# AST
python -m src.grammar_tools.cli ast

# Summary Hard and Soft 
python -m src.grammar_tools.cli validate --policy family
