# evaluation/utils/SelfBleu.py

import os
from multiprocessing import Pool
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from .Metrics import Metrics


class SelfBleu(Metrics):
    def __init__(self, generated_texts: list[str], gram: int = 4):
        super().__init__()
        self.name = 'Self-Bleu'
        # Tokenize the input texts upon initialisation
        self.reference = [nltk.word_tokenize(text.lower()) for text in generated_texts]
        self.gram = gram
        self.sample_size = 500

    def get_score(self, is_fast=True):
        """Calculates the Self-BLEU score for the provided texts."""
        if not self.reference:
            return 0.0
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def calc_bleu(self, reference, hypothesis, weight):
        """Helper to calculate BLEU score for a single hypothesis."""
        return nltk.translate.bleu_score.sentence_bleu(
            reference, hypothesis, weight,
            smoothing_function=SmoothingFunction().method1
        )

    def get_bleu_fast(self):
        """Calculates Self-BLEU on a random sample for speed."""
        # Use a sample of the reference for faster calculation
        sample_ref = self.reference[:self.sample_size]
        return self.get_bleu_parallel(reference=sample_ref)

    def get_bleu_parallel(self, reference=None):
        """Calculates the final Self-BLEU score in parallel."""
        if reference is None:
            reference = self.reference

        ngram = self.gram
        weight = tuple((1. / ngram for _ in range(ngram)))

        # In a small sample, every sentence is a hypothesis
        # and all other sentences are its references.
        pool = Pool(os.cpu_count())
        results = []
        for i in range(len(reference)):
            hypothesis = reference[i]
            other_references = reference[:i] + reference[i + 1:]
            results.append(pool.apply_async(self.calc_bleu, args=(other_references, hypothesis, weight)))

        score = sum(res.get() for res in results)
        pool.close()
        pool.join()

        return score / len(reference) if reference else 0.0