# Generate Synthetic Corpus

## Grammar 1 - run 'loose' corpus generator with config
python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/loose_grammar_run.yaml

## Grammar 2 – run 'balanced' corpus generator with config
python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/balanced_grammar_run.yaml

## Grammar 3 - run 'high' corpus generator with config
python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/high_constraint_grammar_run.yaml


# Convert Jsonl Corpus to Docx
## Looser corpus to docx
python3 src/grammar_tools/utils/convert_dataset_to_docx.py \
        --input data/processed/loose_grammar/loose_500.jsonl \
        --output data/processed/synthetic_session/docx_loose_sessions_dedup

## Balanced corpus to docx 
python3 src/grammar_tools/utils/convert_dataset_to_docx.py \
        --input data/processed/balanced_grammar/balanced_5.jsonl \
        --output data/processed/synthetic_session/docx_sessions_dedup





