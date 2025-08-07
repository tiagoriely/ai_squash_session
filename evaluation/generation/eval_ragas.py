from datasets import load_dataset, Dataset
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas import evaluate
import json

# env
from dotenv import load_dotenv
load_dotenv()


with open("data/eval_dataset.json") as f:
    data = json.load(f)

ragas_dataset = Dataset.from_list(data)

score = evaluate(
    ragas_dataset,
    metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
)

print(score)
