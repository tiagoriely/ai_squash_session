import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import yaml


def run_clustering_analysis(embedding_path: Path, output_path: Path, eps: float, min_samples: int):
    """
    Performs dimensionality reduction and clustering on session embeddings.
    """
    print(f"Loading embeddings from {embedding_path}...")
    # Check if the embedding file actually exists before trying to load it.
    if not embedding_path.exists():
        print(f"⚠️  Error: Embedding file not found at {embedding_path}")
        return None
    embeddings = np.load(embedding_path)

    # 1. Standardise the data - important for DBSCAN
    print("Standardising data...")
    scaled_embeddings = StandardScaler().fit_transform(embeddings)

    # 2. Reduce dimensionality with UMAP for visualisation
    print("Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(scaled_embeddings)

    # 3. Perform DBSCAN clustering on the 2D data
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings_2d)

    # 4. Analyse and report the results
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = np.sum(clusters == -1)
    noise_percent = (n_noise / len(embeddings)) * 100 if embeddings.size > 0 else 0


    print("\n--- Clustering Analysis Report ---")
    print(f"Corpus: {embedding_path.stem}")
    print(f"  - Number of clusters found: {n_clusters}")
    print(f"  - Number of outliers (noise): {n_noise} ({noise_percent:.2f}%)")
    print("----------------------------------")

    # 5. Visualise the clusters
    print("Generating cluster plot...")
    plt.figure(figsize=(12, 10))

    # Create a palette with a specific colour for noise points
    unique_labels = set(clusters)
    palette = sns.color_palette("hsv", len(unique_labels) - (1 if -1 in unique_labels else 0))
    palette_dict = {label: palette[i] for i, label in enumerate(sorted(l for l in unique_labels if l != -1))}
    palette_dict[-1] = (0.5, 0.5, 0.5)  # Grey for noise

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=clusters,
        palette=palette_dict,
        s=50,
        alpha=0.7,
        legend='full'
    )

    plt.title(f"Semantic Clustering of Session Plans ({embedding_path.stem})", fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")


    return {
        "corpus_name": embedding_path.stem,
        "clusters_found": n_clusters,
        "outliers": n_noise,
        "outlier_percent": f"{noise_percent:.2f}%"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DBSCAN clustering on session embeddings from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the clustering_config.yaml file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    ## Get dynamic corpus size from the config. This is the new central variable.
    corpus_size = config['corpus_size']
    print(f"Starting analysis for corpus size: {corpus_size}")

    embedding_base_dir = Path(config.get('embedding_dir', 'evaluation/corpus_analysis/embeddings'))
    output_base_dir = Path(config.get('output_dir', 'evaluation/corpus_analysis/visualisations'))

    # Create a single, dedicated output directory for this run's visualisations and report.
    run_output_dir = output_base_dir / f"corpus_size_{corpus_size}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"➡️  All outputs for this run will be saved in: {run_output_dir}")

    all_results = []
    for task in config.get('clustering_tasks', []):
        ## All path-related strings formatted with the corpus_size.
        task_name = task['name'].format(corpus_size=corpus_size)
        embedding_file = task['embedding_file'].format(corpus_size=corpus_size)
        output_file = task['output_file'].format(corpus_size=corpus_size)

        print(f"\n--- Running Clustering Task: {task_name} ---")

        # Construct the full paths dynamically.
        embedding_path = embedding_base_dir / embedding_file
        # The output path now points inside our dedicated run directory.
        output_path = run_output_dir / output_file
        params = task.get('dbscan_params', {})

        result = run_clustering_analysis(
            embedding_path=embedding_path,
            output_path=output_path,
            eps=params.get('eps', 0.5),
            min_samples=params.get('min_samples', 5)
        )
        # Only append the result if the analysis was successful.
        if result:
            all_results.append(result)

    # Write all collected results to a single CSV
    if all_results:
        report_path = run_output_dir / f"clustering_report_{corpus_size}.csv"
        header = all_results[0].keys()

        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Full clustering report saved to: {report_path}")
    else:
        print("\nNo results were generated. CSV report not created.")