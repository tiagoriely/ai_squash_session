# src/grammar_tools/dedup/jaccard_dedup.py
from __future__ import annotations

import re
from typing import List, Tuple, Set


def _tokenize_for_ngrams(text: str) -> List[str]:
    """
    Lowercase and split into alphanumeric tokens (letters/numbers only).
    Simple/fast; adjust if you need richer tokenization.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_set(tokens: List[str], n: int) -> Set[str]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


class JaccardDeduper:
    """
    Lightweight near-duplicate detector using Jaccard similarity over n-grams
    of the *rendered* text.

    It keeps a rolling window of *accepted* texts (not attempts) and their n-gram sets.
    For a new text, it computes Jaccard similarity vs the window and flags as
    near-duplicate if max similarity >= threshold.

    Args:
        ngram: n for the n-grams (e.g., 2 for bigrams).
        threshold: similarity threshold in [0,1] (>= means near-duplicate).
        window: number of most recent accepted samples to compare against.
                0 means compare against *all* accepted samples.

    Methods:
        is_near_duplicate(text) -> (is_dup, max_sim, argmax_index)
        add(text) -> add the accepted text to the store
        bulk_add(texts) -> add many accepted texts
    """

    def __init__(self, ngram: int = 2, threshold: float = 0.90, window: int = 500):
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if window < 0:
            raise ValueError("window must be >= 0 (0 means compare against all)")

        self.ngram = int(ngram)
        self.threshold = float(threshold)
        self.window = int(window)

        self._texts: List[str] = []
        self._sets: List[Set[str]] = []

    def _make_set(self, text: str) -> Set[str]:
        return _ngram_set(_tokenize_for_ngrams(text), self.ngram)

    def __len__(self) -> int:
        return len(self._texts)

    def is_near_duplicate(self, text: str) -> Tuple[bool, float, int]:
        """
        Returns (is_dup, max_sim, argmax_index).

        argmax_index is the index in the *global* accepted list (self._texts),
        or -1 if store is empty.
        """
        cand = self._make_set(text)
        if not self._sets:
            return (False, 0.0, -1)

        # Select comparison slice
        if self.window > 0:
            compare_sets = self._sets[-self.window :]
            base_idx = len(self._sets) - len(compare_sets)
        else:
            compare_sets = self._sets
            base_idx = 0

        max_sim = -1.0
        max_rel_idx = -1
        for i, s in enumerate(compare_sets):
            sim = jaccard_similarity(cand, s)
            if sim > max_sim:
                max_sim, max_rel_idx = sim, i

        is_dup = max_sim >= self.threshold
        global_idx = base_idx + max_rel_idx if max_rel_idx >= 0 else -1
        return (is_dup, max_sim, global_idx)

    def add(self, text: str) -> None:
        self._texts.append(text)
        self._sets.append(self._make_set(text))

    def bulk_add(self, texts: List[str]) -> None:
        for t in texts:
            self.add(t)
