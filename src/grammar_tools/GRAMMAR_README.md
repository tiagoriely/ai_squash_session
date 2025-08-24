# Generate Synthetic Corpus
## run corpus generator with config
python3 -m src.grammar_tools.grammar_generator_dedup \
  --config configs/synthetic_dataset_creation/balanced_run.yaml

## convert jsonl corpus to docx 
python3 src/grammar_tools/utils/convert_dataset_to_docx.py \
        --input data/processed/balanced_grammar/balanced_5.jsonl \
        --output data/processed/synthetic_session/docx_sessions_dedup
