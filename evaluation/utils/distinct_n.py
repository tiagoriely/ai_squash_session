# evaluation/utils/DistinctnEvaluator.py

import re
import nltk
from nltk.util import ngrams
from collections import Counter
from .Metrics import Metrics


class DistinctnEvaluator(Metrics):
    """
    Calculates Distinct-n metrics to measure lexical diversity.
    A higher score indicates greater diversity.
    """

    def __init__(self):
        super().__init__()
        self.name = 'Distinct-n'

    def _extract_patterns(self, text: str) -> str:
        """
        Uses regex to find and extract hyphen-chained patterns from a generated plan.
        Example: "Conditioned Game: Boast-Cross-Drive (Forehand)" -> "Boast-Cross-Drive"
        """
        # This pattern looks for a colon, optional space, and then captures
        # sequences of words joined by hyphens.
        patterns = re.findall(r":\s*([\w]+(?:-[\w]+)+)", text)
        return " ".join(patterns)

    def _calculate_distinct_n(self, texts: list[str], n: int) -> float:
        """
        Calculates the ratio of unique n-grams to total n-grams.
        """
        if not texts:
            return 0.0

        all_ngrams = []
        tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]

        for tokens in tokenized_texts:
            all_ngrams.extend(ngrams(tokens, n))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)

    def evaluate(self, generated_texts: list[str]) -> dict:
        """
        Calculates the four required Distinct-n metrics for a set of generated plans.

        Args:
            generated_texts (list[str]): A list of generated plan strings from the same query.

        Returns:
            dict: A dictionary containing the four diversity scores.
        """

        # 1. Extract the hyphen-chained patterns from each generated text
        pattern_texts = [self._extract_patterns(text) for text in generated_texts]

        # 2. Calculate the four metrics
        scores = {
            'prose_distinct_1': self._calculate_distinct_n(generated_texts, 1),
            'prose_distinct_2': self._calculate_distinct_n(generated_texts, 2),
            'pattern_distinct_2': self._calculate_distinct_n(pattern_texts, 2),
            'pattern_distinct_3': self._calculate_distinct_n(pattern_texts, 3),
        }

        return scores