# eval_with_individual_scores.py
import json
import pandas as pd
from datetime import datetime
import os

# Import Ragas components
from datasets import Dataset # Used to convert your list of dicts to a Ragas-compatible Dataset
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas import evaluate

# Load environment variables (e.g., your LLM API keys)
from dotenv import load_dotenv
load_dotenv()

# --- NEW: Configuration for the number of samples to evaluate ---
NUM_EVAL_SAMPLES = 5

# --- 1. Load Evaluation Data ---
try:
    with open("data/eval_dataset.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: data/eval_dataset.json not found. Please ensure the file exists.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from data/eval_dataset.json. Check file format.")
    exit()

# --- NEW: Limit the data to the first N samples ---
print(f"✅ Limiting evaluation to the first {NUM_EVAL_SAMPLES} samples from the dataset.")
data = data[:NUM_EVAL_SAMPLES]


# Convert the list of dictionaries to a Ragas Dataset object
ragas_dataset = Dataset.from_list(data)

# --- 2. Perform RAGAS Evaluation ---
print("Starting RAGAS evaluation...")
score_results = evaluate(
    ragas_dataset,
    metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
)
print("RAGAS evaluation complete.")

# --- 3. Prepare Data for Export ---
eval_df = score_results.to_pandas()

overall_scores_dict = {
    "answer_relevancy": eval_df["answer_relevancy"].mean(),
    "context_precision": eval_df["context_precision"].mean(),
    "context_recall": eval_df["context_recall"].mean(),
    "faithfulness": eval_df["faithfulness"].mean(),
}

overall_scores_dict = {k: float(v) for k, v in overall_scores_dict.items()}
individual_results_list = eval_df.to_dict(orient='records')

# --- 4. Define Output Paths ---
output_dir = "evaluation/evaluation_reports/RAGAS"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_full_output_path = os.path.join(output_dir, f"ragas_full_results_{timestamp}.json")
excel_summary_output_path = os.path.join(output_dir, f"ragas_summary_report_{timestamp}.xlsx")
json_overall_output_path = os.path.join(output_dir, f"ragas_overall_scores_{timestamp}.json")

# --- 5. Export Results ---
with open(json_full_output_path, "w") as f:
    json.dump(individual_results_list, f, indent=4)
print(f"Full RAGAS results (JSON) saved to: {json_full_output_path}")

eval_df.to_excel(excel_summary_output_path, index=False)
print(f"Summary RAGAS report (Excel) saved to: {excel_summary_output_path}")

with open(json_overall_output_path, "w") as f:
    json.dump(overall_scores_dict, f, indent=4)
print(f"Overall RAGAS scores (JSON) saved to: {json_overall_output_path}")


# --- 6. Print Overall Scores to Console (for immediate feedback) ---
print("\n--- Overall RAGAS Scores ---")
print(score_results)