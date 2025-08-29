
# Generate Queries
```bash
$ python3 -m evaluation.query_sets.generators.01_generate_out_of_dist
$ python3 -m evaluation.query_sets.generators.03_generate_complexity
```

⚠️ Pick the number of queries and the grammar type and size in config file
```bash
$ python3 -m evaluation.query_sets.generators.02_generate_golden_set balanced_grammar
```


# Evaluate Retrievers

⚠️⚠️go to sparse and dense config and pick the size and type of grammar ⚠️⚠️
$ python -m evaluation.retrieval.evaluate_retrievers <grammar_type>

# Analyse Retrievers' Results
```bash
python3 -m evaluation.analyse_results 
```


