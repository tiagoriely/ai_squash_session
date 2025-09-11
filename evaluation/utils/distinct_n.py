# evaluation/utils/DistinctnEvaluator.py

import re
import nltk
from nltk.util import ngrams
from typing import List, Dict

# We import SQUASH_PHRASES to identify specific shots in the generated text
from rag.utils import SQUASH_PHRASES
from .Metrics import Metrics


class DistinctnEvaluator(Metrics):
    """
    Calculates a suite of Distinct-n metrics to measure lexical diversity,
    separating prose from structured exercise patterns and shots.
    """

    def __init__(self):
        super().__init__()
        self.name = 'Distinct-n'
        # A pre-compiled regex for finding exercise patterns
        self.pattern_regex = re.compile(r":\s*([\w]+(?:-[\w]+)+)")
        # A pre-compiled regex to find and remove the patterns for prose extraction
        self.prose_filter_regex = re.compile(r":\s*[\w]+(?:-[\w]+)+")

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Helper to generate n-grams from a list of tokens."""
        return list(ngrams(tokens, n))

    def _calculate_distinct_score(self, ngrams_list: List[tuple]) -> float:
        """Calculates the ratio of unique n-grams to total n-grams."""
        if not ngrams_list:
            return 0.0
        return len(set(ngrams_list)) / len(ngrams_list)

    def evaluate(self, generated_texts: List[str]) -> Dict[str, float]:
        """
        Calculates four distinct diversity metrics for a set of generated plans.
        """
        # --- Data Preparation ---
        all_prose_tokens = []
        all_pattern_tokens = []
        all_specific_shot_tokens = []

        # Sort phrases by length to match longer phrases first
        sorted_specific_shots = sorted(SQUASH_PHRASES, key=len, reverse=True)

        for text in generated_texts:
            # 1. Extract the structured patterns
            patterns = self.pattern_regex.findall(text)
            all_pattern_tokens.extend(patterns)

            # 2. Extract the prose by removing patterns from the original text
            prose_text = self.prose_filter_regex.sub("", text)
            all_prose_tokens.extend(nltk.word_tokenize(prose_text.lower()))

            # 3. Extract specific shots from the original text
            text_lower = text.lower()
            for shot in sorted_specific_shots:
                if shot in text_lower:
                    # Replace spaces with underscores to treat as a single token
                    shot_token = shot.replace(" ", "_")
                    all_specific_shot_tokens.append(shot_token)

        # --- Metric Calculation ---
        # Prose Metrics
        prose_ngrams_1 = self._get_ngrams(all_prose_tokens, 1)
        prose_ngrams_2 = self._get_ngrams(all_prose_tokens, 2)

        # Pattern Metrics (Distinct-1 on the patterns themselves)
        pattern_ngrams_1 = self._get_ngrams(all_pattern_tokens, 1)

        # Specific Shot Metrics
        specific_shot_ngrams_2 = self._get_ngrams(all_specific_shot_tokens, 2)

        scores = {
            'prose_distinct_1': self._calculate_distinct_score(prose_ngrams_1),
            'prose_distinct_2': self._calculate_distinct_score(prose_ngrams_2),
            'pattern_diversity': self._calculate_distinct_score(pattern_ngrams_1),
            'specific_shot_diversity_2': self._calculate_distinct_score(specific_shot_ngrams_2),
        }

        return scores