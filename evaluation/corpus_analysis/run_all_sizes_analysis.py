import argparse
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing analysis functions
from evaluation.corpus_analysis.statistics.measure_diversity import analyse_diversity_metrics
from evaluation.corpus_analysis.statistics.measure_structure import analyse_structure_metrics
from evaluation.corpus_analysis.statistics.measure_reliability import analyse_reliability_metrics
from evaluation.corpus_analysis.statistics.pillar_scores import compute_pillar_scores
from evaluation.corpus_analysis.utils import load_corpus


def plot_results(results_df: pd.DataFrame, output_path: Path):
    """Generates and saves a line plot of impact score vs. corpus size."""
    print(f"\nGenerating plot from {len(results_df)} data points...")

    # Set a professional plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot the data using seaborn for easier legend handling
    plot = sns.lineplot(
        data=results_df,
        x="size",
        y="overall_impact",
        hue="grammar",
        style="grammar",
        markers=True,
        dashes=False,
        palette="viridis",
        linewidth=2.5
    )

    # Set plot titles and labels
    plt.title("Overall Impact Score vs. Corpus Size", fontsize=16, weight='bold')
    plt.xlabel("Corpus Size", fontsize=12)
    plt.ylabel("Overall Impact Score", fontsize=12)
    plt.legend(title="Grammar Constraint")

    # Set axis limits to provide better focus
    plt.ylim(0.85, 1.0)
    plt.xticks(results_df['size'].unique())  # Ensure all tested sizes are marked

    # Save the plot
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"✅ Plot successfully saved to: {output_path}")


def main():
    """Main function to run analysis and generate plots."""
    parser = argparse.ArgumentParser(description="Run full corpus analysis and plot results.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the analysis config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Config reading ---
    corpus_sizes = config['corpus_sizes']
    input_dir = Path(config['input_data_dir'])
    profiles = config['grammar_profiles']
    plot_output_path = Path(config['plot_output_path'])

    all_results_data = []

    # --- Loop over all specified corpus sizes and profiles ---
    for size in corpus_sizes:
        print("-" * 60)
        print(f"Analysing corpus size: {size}...")
        for profile in profiles:
            profile_name = profile['name']
            corpus_path = input_dir / profile_name / f"{profile['adjective']}_{size}.jsonl"

            if not corpus_path.exists():
                print(f"⚠️  Warning: Corpus file not found at {corpus_path}. Skipping.")
                continue

            corpus = load_corpus(corpus_path)
            if not corpus:
                print(f"⚠️  Warning: Corpus at {corpus_path} is empty. Skipping.")
                continue

            # Run all analyses
            diversity = analyse_diversity_metrics(corpus, grammar_profile=profile_name)
            structure = analyse_structure_metrics(corpus)
            reliability = analyse_reliability_metrics(corpus)

            # Compute final pillar scores
            scores = compute_pillar_scores(diversity, structure, reliability)

            # Append the key results for plotting
            all_results_data.append({
                "size": size,
                "grammar": profile['plot_label'],
                "overall_impact": scores.get("overall_impact", 0.0),
                **scores
            })

    if not all_results_data:
        print("No results were generated. Exiting before plotting.")
        return

    # Convert results to a pandas DataFrame for easy plotting
    results_df = pd.DataFrame(all_results_data)

    # Generate and save the plot
    plot_results(results_df, plot_output_path)

    # Optional: save the full dataframe to a single CSV for inspection
    csv_output_path = plot_output_path.parent / "full_analysis_results.csv"
    results_df.to_csv(csv_output_path, index=False)
    print(f"💾 Full numerical results saved to: {csv_output_path}")


if __name__ == "__main__":
    main()