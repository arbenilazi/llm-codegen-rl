from __future__ import annotations

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL: SentenceTransformer | None = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_embeddings(codes: List[str]) -> np.ndarray:
    """
    Compute sentence-transformer embeddings for a list of code snippets.
    """
    if not codes:
        return np.empty((0, 384), dtype=np.float32)
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME, device=_device())
    embeddings = _MODEL.encode(codes, convert_to_numpy=True, normalize_embeddings=False)
    return embeddings


def pairwise_distances(emb: np.ndarray) -> np.ndarray:
    """
    Cosine distance matrix for embeddings.
    """
    if emb.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = emb / norms
    similarity = np.clip(np.matmul(normalized, normalized.T), -1.0, 1.0)
    distances = 1.0 - similarity
    return distances
