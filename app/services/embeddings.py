"""Generating embeddings (Week 3).

An *embedding* turns a text into a vector of numbers that captures its meaning:
texts with similar meaning produce nearby vectors. This is what makes "semantic"
search possible (by meaning, not by exact words).

We use the `all-MiniLM-L6-v2` model: small, fast, runs on CPU and produces
384-dimensional vectors.

Two interchangeable backends run the same model (chosen with EMBEDDING_BACKEND):
- "fastembed" (default): runs the model with ONNX Runtime. Light, no PyTorch,
  works on Windows machines where Smart App Control blocks PyTorch's DLLs.
- "sentence-transformers": runs the model with PyTorch.

The rest of the app only calls `embed_texts` / `embed_query`, so switching
backend never touches the routers or the database. Documents never leave the
machine with either backend.

Technical notes:
- the model is loaded ONCE (it is expensive) and reused;
- vectors are normalized to unit length, which makes the cosine distance
  stable and comparable;
- documents and questions MUST be embedded with the same model, otherwise
  their vectors are not comparable (changing model means re-indexing).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import Settings

SUPPORTED_BACKENDS = ("fastembed", "sentence-transformers")


@lru_cache(maxsize=2)
def _load_fastembed(model_name: str, cache_dir: str):
    # Local import: the app can be imported (and most tests run) without
    # loading the model library.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


@lru_cache(maxsize=2)
def _load_sentence_transformers(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero on an all-zero vector
    return vectors / norms


def embed_texts(texts: list[str], settings: Settings) -> list[np.ndarray]:
    """Turn a list of texts into a list of unit-length float32 vectors."""
    texts = list(texts)
    if not texts:
        return []

    backend = settings.embedding_backend
    if backend == "fastembed":
        model = _load_fastembed(settings.embedding_model_name, settings.embedding_cache_dir)
        matrix = np.array(list(model.embed(texts)), dtype=np.float32)
    elif backend == "sentence-transformers":
        model = _load_sentence_transformers(settings.embedding_model_name)
        matrix = np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND {backend!r}; use one of {SUPPORTED_BACKENDS}"
        )

    if matrix.shape[1] != settings.embedding_dim:
        # A model with a different size would not fit the vector(N) DB column.
        raise ValueError(
            f"Model produced {matrix.shape[1]}-dim vectors, "
            f"but EMBEDDING_DIM is {settings.embedding_dim}"
        )

    return list(_normalize(matrix))


def embed_query(text: str, settings: Settings) -> np.ndarray:
    """Convenience: embedding of a single text (the user's question)."""
    return embed_texts([text], settings)[0]
