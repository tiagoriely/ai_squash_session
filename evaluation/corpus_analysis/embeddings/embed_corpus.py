# evaluation/corpus_analysis/_1_embedding/embed_corpus.py

import argparse
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from evaluation.corpus_analysis.utils import load_corpus


def create_embeddings(
        corpus_path: Path,
        output_path: Path,
        model_name: str,
        device: str | None
):
    """
    Generates semantic embeddings for each session in a corpus and saves them.
    """
    corpus = load_corpus(corpus_path)
    if not corpus:
        return

    # Extract the text content from each session document
    session_texts = [session.get('contents', '') for session in corpus]

    print(f"Loading sentence transformer model: '{model_name}'...")
    # Determine the device (e.g., 'cuda' for GPU, 'cpu', or auto)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    print(f"Embedding {len(session_texts)} sessions... (This may take some time)")
    embeddings = model.encode(
        session_texts,
        show_progress_bar=True,
        normalize_embeddings=True  # Normalising is good practice for similarity/clustering tasks
    )

    # Ensure the output directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Save the embeddings as a NumPy array
    np.save(output_path, embeddings)
    print(f"✅ Embeddings saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save semantic embeddings for a session corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the .jsonl corpus file.")
    parser.add_argument("output_path", type=Path, help="Path to save the output .npy embedding file.")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda'). Defaults to auto-detection."
    )
    args = parser.parse_args()

    create_embeddings(args.corpus_path, args.output_path, args.model, args.device)