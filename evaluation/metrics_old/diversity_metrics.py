# evaluation/metrics/diversity_metrics.py

import re
import itertools
from typing import List
from numpy import mean

# It's good practice to handle potential import errors if run standalone
try:
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
except ImportError:
    print("Warning: sentence-transformers or scipy not found. IPSD metric will not be available.")
    SentenceTransformer = None
    cosine = None


def calculate_ipv(generated_plan_text: str) -> int:
    """
    Calculates Intra-Plan Variety (IPV).

    Counts the number of unique, named exercises in a generated session plan.

    Args:
        generated_plan_text (str): The full text of the generated session plan.

    Returns:
        int: The number of unique exercises.
    """
    # Regex to capture the exercise name after "Drill:" or "Conditioned Game:"
    exercise_names = re.findall(r":\s*(?:Drill|Conditioned Game):\s*(.*?)\s*(?:\(|$)", generated_plan_text)

    # Use a set to count unique names, stripping any trailing whitespace
    unique_exercises = {name.strip() for name in exercise_names}

    return len(unique_exercises)


def calculate_ipsd(
        list_of_plan_texts: List[str],
        embedding_model: any,  # Expects a SentenceTransformer model instance
) -> float:
    """
    Calculates Inter-Plan Semantic Distance (IPSD).

    Measures the average semantic distance (cosine distance) between all pairs
    of generated plans for the same query. A higher score means more diverse outputs.

    Args:
        list_of_plan_texts (List[str]): A list of N generated plan strings.
        embedding_model (Any): An initialized sentence-transformer model.

    Returns:
        float: The average cosine distance, or 0.0 if fewer than 2 plans are provided.
    """
    if SentenceTransformer is None or cosine is None:
        raise ImportError("sentence-transformers and scipy are required for IPSD calculation.")

    if len(list_of_plan_texts) < 2:
        return 0.0

    # Generate embeddings for all plan texts
    embeddings = embedding_model.encode(list_of_plan_texts, show_progress_bar=False)

    # Calculate cosine distance for all unique pairs of embeddings
    all_pairs = list(itertools.combinations(embeddings, 2))
    distances = [cosine(pair[0], pair[1]) for pair in all_pairs]

    return mean(distances) if distances else 0.0