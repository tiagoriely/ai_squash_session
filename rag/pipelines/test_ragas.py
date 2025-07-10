import pytest
ragas = pytest.importorskip("ragas")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv
load_dotenv()

# Example data
data = {
    "user_input": ["What is the capital of France?"],
    "response": ["Paris is the capital of France."],
    "retrieved_contexts": [["Paris is the capital of France. It is a major European city known for its culture."]],
    "reference": ["Paris"]
}

# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Define the metrics you want to evaluate
metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
]

# Evaluate the dataset using the selected metrics
results = evaluate(dataset, metrics)

# Evaluate
results = evaluate(dataset, metrics)

# Grab the dict for example #0
example0_scores = results.scores[0]

# Now iterate cleanly
for metric_name, score in example0_scores.items():
    print(f"{metric_name}: {score:.2f}")

