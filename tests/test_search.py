"""End-to-end test for Week 3: indexing + semantic search.

This test is "heavy": it needs the embedding libraries and a running
Postgres+pgvector. If anything is missing (libraries, model, or DB), it is
SKIPPED (it does not fail), so the suite stays green on a machine without a DB,
without the models downloaded, or where PyTorch cannot load.

Start the DB first:  docker compose up -d
"""

import io

import pytest

# The embedding backend may be unavailable (not installed, or its native
# libraries blocked by the OS). We skip the whole file instead of failing.
try:
    import sentence_transformers  # noqa: F401

    _EMBEDDINGS_OK = True
except Exception:  # ImportError, OSError (blocked DLLs), etc.
    _EMBEDDINGS_OK = False

pytest.importorskip("pgvector")
pymupdf = pytest.importorskip("pymupdf")
import psycopg  # noqa: E402

from app.config import Settings  # noqa: E402


def _db_available() -> bool:
    try:
        psycopg.connect(Settings().database_url, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _EMBEDDINGS_OK, reason="Embedding backend unavailable"),
    pytest.mark.skipif(not _db_available(), reason="Postgres/pgvector unavailable"),
]


# Two pages on clearly different topics: so the search has a clear winner.
PAGE_CATS = (
    "Cats are affectionate pets. "
    "They love to sleep, play, and their purring relaxes whoever pets them. "
) * 8
PAGE_CARS = (
    "The internal combustion engine powers many automobiles. "
    "Maintaining the gearbox and the brakes is important for safety. "
) * 8


def _make_pdf(*pages_text: str) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 500, 760), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_index_and_search(client):
    # 1) Upload a 2-page PDF (cats / cars).
    pdf = _make_pdf(PAGE_CATS, PAGE_CARS)
    up = client.post("/documents", files={"file": ("doc.pdf", io.BytesIO(pdf), "application/pdf")})
    assert up.status_code == 201, up.text
    document_id = up.json()["document_id"]

    # 2) Index (extraction -> chunk -> embeddings -> DB).
    idx = client.post(f"/documents/{document_id}/index")
    assert idx.status_code == 200, idx.text
    assert idx.json()["chunks_indexed"] > 0

    # 3) Semantic search with a question about cats, WITHOUT using the same words.
    res = client.get("/search", params={"q": "which animals purr?", "top_k": 3})
    assert res.status_code == 200, res.text
    results = res.json()["results"]
    assert results, "the search returned no results"

    # The best result must come from the cats page (page 1).
    assert results[0]["page_number"] == 1
    # And the score (cosine similarity) must be a sensible number.
    assert 0.0 < results[0]["score"] <= 1.0

    # Cleanup: remove this document's chunks from the dev DB.
    conn = psycopg.connect(Settings().database_url)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    conn.commit()
    conn.close()
