"""Shared helpers for loading GloVe vectors (used by load.py and app.py)."""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_NAME = "glove_words"


def glove_path(dim):
    return os.path.join(DATA_DIR, f"glove.6B.{dim}d.txt")


def load_vectors(dim, limit=None):
    """Load GloVe vectors, L2-normalized so inner-product search == cosine similarity.

    GloVe files are ordered by descending word frequency, so `limit` keeps the
    most common words. Returns (words, vectors) where vectors is float32 [N, dim].
    """
    path = glove_path(dim)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python download_data.py --dim {dim}"
        )

    words = []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))

    vectors = np.vstack(rows)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    return words, vectors
