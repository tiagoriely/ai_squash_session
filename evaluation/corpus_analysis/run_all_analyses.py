# evaluation/corpus_analysis/run_all_analyses.py
import argparse
import csv
from pathlib import Path
import json
import yaml

from evaluation.corpus_analysis.statistics.measure_diversity import analyse_diversity_metrics
from evaluation.corpus_analysis.statistics.measure_structure import analyse_structure_metrics
from evaluation.corpus_analysis.statistics.measure_reliability import analyse_reliability_metrics


def run_full_analysis(profiles_to_analyse: list, corpus_size: int, input_base_dir: Path, output_csv_path: Path):
    """
    Runs all statistical analyses on all specified grammar corpora and saves a summary CSV.
    """
    all_results = []

    for profile_info in profiles_to_analyse:
        profile_name = profile_info['name']
        adjective = profile_info['adjective']

        print(f"\nAnalysing corpus for: {profile_name} (size: {corpus_size})...")
        corpus_path = input_base_dir / profile_name / f"{adjective}_{corpus_size}.jsonl"

        if not corpus_path.exists():
            print(f"⚠️  Warning: Corpus file not found at {corpus_path}. Skipping.")
            continue

        diversity_results = analyse_diversity_metrics(corpus_path, profile_name)
        structure_results = analyse_structure_metrics(corpus_path)
        reliability_results = analyse_reliability_metrics(corpus_path, profile_name)

        flat_results = {"grammar_profile": profile_name, **diversity_results, **structure_results,
                        **reliability_results}
        all_results.append(flat_results)

    # --- Write to CSV ---
    if not all_results:
        print("No results to write.")
        return

    output_csv_path.parent.mkdir(exist_ok=True, parents=True)
    header = all_results[0].keys()

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in all_results:
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
            writer.writerow(row)

    print(f"\n✅ Full analysis report saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all corpus analyses from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the analysis_config.yaml file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- config reading ---
    corpus_size = config['corpus_size']
    input_dir = Path(config['input_data_dir'])
    # Get the BASE output directory, not the final one.
    output_base_dir = Path(config['output_dir'])
    profiles = config['grammar_profiles']

    # This ensures all results for a given size are co-located.
    run_output_dir = output_base_dir / f"corpus_size_{corpus_size}"

    # Define the final output file path within that shared directory.
    output_csv_path = run_output_dir / f"stats_analysis_report_{corpus_size}.csv"

    print(f"Starting full analysis for corpus size: {corpus_size}")
    print(f"➡️  Analysis report will be saved to: {output_csv_path}")

    # Pass the correctly constructed path to the main function ---
    run_full_analysis(profiles, corpus_size, input_dir, output_csv_path)