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

# --- 1. Load Evaluation Data ---
# Ensure your eval_dataset.json is correctly structured as a list of dictionaries,
# where each dictionary has at least 'question', 'answer', 'contexts', and 'reference' (even if empty string)
try:
    with open("data/eval_dataset.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: data/eval_dataset.json not found. Please ensure the file exists.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from data/eval_dataset.json. Check file format.")
    exit()

# Convert the list of dictionaries to a Ragas Dataset object
ragas_dataset = Dataset.from_list(data)

# --- 2. Perform RAGAS Evaluation ---
print("Starting RAGAS evaluation...")
score_results = evaluate(
    ragas_dataset,
    metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
    # You can add 'llm' and 'embeddings' parameters here if you want to explicitly
    # define the models Ragas uses for evaluation, overriding defaults.
    # e.g., llm=RagasLLM(model="google/gemini-1.5-pro"),
    # embeddings=RagasEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    # Make sure relevant libraries are installed and API keys are set if using commercial models.
)
print("RAGAS evaluation complete.")

# --- 3. Prepare Data for Export ---

# Convert the Ragas EvaluationResult object to a pandas DataFrame.
# This DataFrame will contain all original columns from your dataset plus
# new columns for each Ragas metric score per row.
eval_df = score_results.to_pandas()

# Calculate overall (mean) scores for each metric
# .mean() will correctly handle NaN values (e.g., if context_recall couldn't be calculated)
overall_scores_dict = {
    "answer_relevancy": eval_df["answer_relevancy"].mean(),
    "context_precision": eval_df["context_precision"].mean(),
    "context_recall": eval_df["context_recall"].mean(),
    "faithfulness": eval_df["faithfulness"].mean(),
}

# Ensure all mean values are standard Python floats for JSON serialization
overall_scores_dict = {k: float(v) for k, v in overall_scores_dict.items()}

# Get individual results as a list of dictionaries (preserving all columns)
individual_results_list = eval_df.to_dict(orient='records')

# --- 4. Define Output Paths ---
output_dir = "evaluation_reports"
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

# Generate a unique timestamp for the filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_full_output_path = os.path.join(output_dir, f"ragas_full_results_{timestamp}.json")
excel_summary_output_path = os.path.join(output_dir, f"ragas_summary_report_{timestamp}.xlsx")
json_overall_output_path = os.path.join(output_dir, f"ragas_overall_scores_{timestamp}.json")

# --- 5. Export Results ---

# Export full individual results to a JSON file
# This will overwrite any file with the exact same name (due to timestamp, usually unique)
with open(json_full_output_path, "w") as f:
    json.dump(individual_results_list, f, indent=4)
print(f"Full RAGAS results (JSON) saved to: {json_full_output_path}")

# Export summary report to an Excel file
# This will also overwrite any file with the exact same name
eval_df.to_excel(excel_summary_output_path, index=False)
print(f"Summary RAGAS report (Excel) saved to: {excel_overall_output_path}")

# Export overall aggregated scores to a separate JSON file
# This will also overwrite any file with the exact same name
with open(json_overall_output_path, "w") as f:
    json.dump(overall_scores_dict, f, indent=4)
print(f"Overall RAGAS scores (JSON) saved to: {json_overall_output_path}")


# --- 6. Print Overall Scores to Console (for immediate feedback) ---
print("\n--- Overall RAGAS Scores ---")
print(score_results) # This prints the nice formatted table from Ragas