# src/dedup/semantic_dedup.py
from __future__ import annotations

from typing import Optional, Tuple, List

class SemanticDedupNotAvailable(RuntimeError):
    """Raised when required libraries for semantic deduplication are missing."""
    pass


class SemanticDeduper:
    """
    Near-duplicate detector using sentence embeddings + cosine similarity.

    - Encodes texts with a SentenceTransformer (default: all-MiniLM-L6-v2).
    - Keeps an in-memory, L2-normalized embedding matrix.
    - For a new text, computes max cosine similarity to existing embeddings.
    - If max_sim >= threshold, treat as near-duplicate.

    Notes:
      * O(N) similarity per insert. Fine up to ~50k. For larger corpora,
        consider FAISS/ANN, sharding, or periodic clustering.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        threshold: float = 0.95,
        dtype: str = "float32",
    ) -> None:
        try:
            import numpy as np  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as e:
            raise SemanticDedupNotAvailable(
                "Semantic dedup requires 'sentence-transformers' and 'numpy'. "
                "Install with: pip install sentence-transformers numpy"
            ) from e

        import numpy as _np
        from sentence_transformers import SentenceTransformer as _ST

        self._np = _np
        self.normalize = normalize
        self.threshold = float(threshold)
        self._dtype = _np.float32 if dtype == "float32" else _np.float64

        # Load model once
        self._model = _ST(model_name, device=device) if device else _ST(model_name)

        # Embedding storage
        self._embs: Optional[_np.ndarray] = None  # shape (N, D)
        self._dim: Optional[int] = None
        self._count: int = 0

    def _encode(self, texts: List[str]) -> "self._np.ndarray":
        vecs = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize manually for consistency
            show_progress_bar=False,
        ).astype(self._dtype, copy=False)

        if self.normalize:
            norms = self._np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs

    def _ensure_initialized(self, dim: int) -> None:
        if self._embs is None:
            self._embs = self._np.zeros((0, dim), dtype=self._dtype)
            self._dim = dim

    def is_near_duplicate(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, int]:
        """
        Returns (is_dup, max_sim, argmax_index). If store is empty, returns (False, 0.0, -1).
        """
        thr = self.threshold if threshold is None else float(threshold)
        vec = self._encode([text])[0]

        if self._embs is None or self._embs.shape[0] == 0:
            return (False, 0.0, -1)

        sims = self._embs @ vec  # cosine if normalized
        idx = int(sims.argmax())
        max_sim = float(sims[idx])
        return (max_sim >= thr, max_sim, idx)

    def add(self, text: str) -> None:
        vec = self._encode([text])[0]
        self._ensure_initialized(dim=vec.shape[0])
        self._embs = self._np.vstack([self._embs, vec])  # type: ignore[arg-type]
        self._count += 1

    def bulk_add(self, texts: List[str]) -> None:
        vecs = self._encode(texts)
        self._ensure_initialized(dim=vecs.shape[1])
        if self._embs is None or self._embs.shape[0] == 0:
            self._embs = vecs
        else:
            self._embs = self._np.vstack([self._embs, vecs])  # type: ignore[arg-type]
        self._count += vecs.shape[0]

    def __len__(self) -> int:
        return self._count
