"""Unit tests for the embedding service (Week 3).

They do not need the database. They use the fastembed backend, which downloads
the model on first run (~90 MB) and then reuses it from EMBEDDING_CACHE_DIR.
Skipped if fastembed is not installed.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from app.config import Settings

pytest.importorskip("fastembed")

from app.services.embeddings import embed_query, embed_texts  # noqa: E402


@pytest.fixture(scope="module")
def emb_settings() -> Settings:
    return Settings(embedding_backend="fastembed")


def test_vectors_have_expected_size_and_unit_length(emb_settings):
    vectors = embed_texts(["first text", "second text"], emb_settings)

    assert len(vectors) == 2
    for v in vectors:
        assert v.shape == (emb_settings.embedding_dim,)
        assert v.dtype == np.float32
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_similar_meaning_scores_higher(emb_settings):
    question = embed_query("Which animals purr?", emb_settings)
    cats, engine = embed_texts(
        ["Cats purr when they are happy.", "The car engine needs oil."], emb_settings
    )

    # Vectors are unit length, so the dot product is the cosine similarity.
    assert float(question @ cats) > float(question @ engine)


def test_empty_input_returns_empty_list(emb_settings):
    assert embed_texts([], emb_settings) == []


def test_unknown_backend_is_rejected_by_config():
    with pytest.raises(ValidationError):
        Settings(embedding_backend="not-a-backend")


def test_wrong_dimension_is_detected():
    # The model gives 384 numbers; a mismatching EMBEDDING_DIM must fail loudly
    # instead of breaking later at the database insert.
    settings = Settings(embedding_backend="fastembed", embedding_dim=768)
    with pytest.raises(ValueError, match="384-dim"):
        embed_texts(["hello"], settings)
