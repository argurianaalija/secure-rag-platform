"""Tests for the upload endpoint.

They cover the happy path and the main error cases. These tests are the Week 1
"deliverable" put to the test: they show that the API accepts valid PDFs and
rejects everything else in a predictable way.
"""

import io

# A minimal but valid PDF: it starts with the magic bytes "%PDF-".
MINIMAL_PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_valid_pdf(client):
    files = {"file": ("doc.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")}
    resp = client.post("/documents", files=files)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["filename"] == "doc.pdf"
    assert body["size_bytes"] == len(MINIMAL_PDF)
    assert body["content_type"] == "application/pdf"
    assert body["document_id"]


def test_rejects_non_pdf(client):
    # .pdf extension but content that is NOT a PDF: must be rejected.
    files = {"file": ("fake.pdf", io.BytesIO(b"this is not a pdf"), "application/pdf")}
    resp = client.post("/documents", files=files)

    assert resp.status_code == 422
    assert resp.json()["error"] == "corrupted_file"


def test_rejects_empty_file(client):
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    resp = client.post("/documents", files=files)

    assert resp.status_code == 422
    assert resp.json()["error"] == "empty_file"


def test_rejects_file_too_large(client):
    # Limit in tests = 1 MB. Build a "PDF" of ~2 MB.
    big = MINIMAL_PDF + b"0" * (2 * 1024 * 1024)
    files = {"file": ("big.pdf", io.BytesIO(big), "application/pdf")}
    resp = client.post("/documents", files=files)

    assert resp.status_code == 413
    assert resp.json()["error"] == "file_too_large"


def test_missing_file_field(client):
    resp = client.post("/documents", data={})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"
