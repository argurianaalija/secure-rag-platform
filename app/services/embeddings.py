"""Generating embeddings with Sentence Transformers (Week 3).

An *embedding* turns a text into a vector of numbers that captures its meaning:
texts with similar meaning produce nearby vectors. This is what makes "semantic"
search possible (by meaning, not by exact words).

We use the `all-MiniLM-L6-v2` model: small, fast, runs on CPU and produces
384-dimensional vectors. Perfect for learning and for a portfolio project.

Technical note: we load the model ONCE (it is expensive) and reuse it;
`normalize_embeddings=True` returns unit-length vectors, which makes the cosine
distance stable and comparable.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import Settings


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load (and cache) the embedding model.

    The import is local so the app can be imported even where sentence-
    transformers is not needed (e.g. some tests), without paying its cost.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], settings: Settings) -> list[np.ndarray]:
    """Turn a list of texts into a list of vectors (float32)."""
    model = _load_model(settings.embedding_model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_query(text: str, settings: Settings) -> np.ndarray:
    """Convenience: embedding of a single text (the user's question)."""
    return embed_texts([text], settings)[0]
