import argparse
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

    print("\n--- Clustering Analysis Report ---")
    print(f"Corpus: {embedding_path.stem}")
    print(f"  - Number of clusters found: {n_clusters}")
    print(f"  - Number of outliers (noise): {n_noise} ({n_noise / len(embeddings) * 100:.2f}%)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DBSCAN clustering on session embeddings from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the clustering_config.yaml file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    embedding_base_dir = Path(config.get('embedding_dir', 'evaluation/corpus_analysis/embeddings'))
    output_base_dir = Path(config.get('output_dir', 'evaluation/corpus_analysis/visualisations'))

    for task in config.get('clustering_tasks', []):
        print(f"\n--- Running Clustering Task: {task['name']} ---")

        embedding_path = embedding_base_dir / task['embedding_file']
        output_path = output_base_dir / task['output_file']
        params = task.get('dbscan_params', {})

        run_clustering_analysis(
            embedding_path=embedding_path,
            output_path=output_path,
            eps=params.get('eps', 0.5),
            min_samples=params.get('min_samples', 5)
        )