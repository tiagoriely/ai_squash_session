# evaluation/corpus_analysis/_3_statistics/common_metrics.py
import math
from typing import List

def calculate_shannon_entropy(probabilities: List[float]) -> float:
    """Calculates the Shannon Entropy for a given distribution of probabilities."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy