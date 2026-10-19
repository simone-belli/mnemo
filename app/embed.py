"""Embedding interface. Anything with `name` and `embed(texts) -> (n, d) unit vectors` will do."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class SentenceTransformerEmbedder:
    """Local model via sentence-transformers (default `BAAI/bge-small-en-v1.5`). Loaded on first use."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class HashEmbedder:
    """Bag of hashed word stems. No downloads and no semantics: it only matches shared words,
    which is enough for tests and for trying the app offline."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _index(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) < 3:
                    continue
                out[i, self._index(word[:5])] += 1.0
        return _normalise(out)


def make_embedder(model_name: str) -> Embedder:
    return HashEmbedder() if model_name == "hash" else SentenceTransformerEmbedder(model_name)
