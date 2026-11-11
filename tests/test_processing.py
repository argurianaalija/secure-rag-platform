"""End-to-end test for Week 2: from upload to chunks.

We generate a real 2-page PDF on the fly (with PyMuPDF), upload it through the
API and then request its chunks. This verifies extraction, chunking and the
endpoint together.
"""

import io

import pymupdf

PAGE_1 = ("This is the text of the first page. " * 20).strip()
PAGE_2 = ("The content of the second page is different. " * 20).strip()


def _make_pdf(*pages_text: str) -> bytes:
    """Create an in-memory PDF with one page per given text."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 500, 760), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client, pdf_bytes: bytes) -> str:
    files = {"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    resp = client.post("/documents", files=files)
    assert resp.status_code == 201, resp.text
    return resp.json()["document_id"]


def test_chunks_from_real_pdf(client):
    document_id = _upload(client, _make_pdf(PAGE_1, PAGE_2))

    resp = client.get(f"/documents/{document_id}/chunks")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["document_id"] == document_id
    assert body["page_count"] == 2
    assert body["chunk_count"] > 0
    assert len(body["chunks"]) == body["chunk_count"]

    # Each chunk has text and a valid page (1 or 2).
    for chunk in body["chunks"]:
        assert chunk["text"].strip()
        assert chunk["page_number"] in (1, 2)

    # Chunks from both pages must appear.
    pages = {c["page_number"] for c in body["chunks"]}
    assert pages == {1, 2}


def test_chunks_missing_document(client):
    # Formally valid id (32 hex chars) but non-existent -> 404.
    resp = client.get("/documents/" + "a" * 32 + "/chunks")
    assert resp.status_code == 404
    assert resp.json()["error"] == "document_not_found"


def test_chunks_malformed_document_id(client):
    # Invalid id (contains non-hex characters / wrong length).
    resp = client.get("/documents/not-valid/chunks")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_document_id"
