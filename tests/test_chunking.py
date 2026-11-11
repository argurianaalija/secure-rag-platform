"""Unit tests for the chunking logic (without PDFs).

We verify the properties that matter for a good RAG: each chunk knows which page
it came from, indices are progressive, and there is real overlap between
consecutive blocks of the same page.
"""

import pytest

from app.services.chunking import chunk_document
from app.services.extraction import ExtractedDocument, PageText


def _extracted(*pages_text: str) -> ExtractedDocument:
    pages = [PageText(page_number=i, text=t) for i, t in enumerate(pages_text, start=1)]
    return ExtractedDocument(pages=pages)


def test_chunk_keeps_page_and_indices():
    extracted = _extracted("a" * 250, "b" * 120)
    chunks = chunk_document("doc", extracted, chunk_size=100, overlap=20)

    # Progressive indices with no gaps.
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # Each chunk knows its page; pages are only 1 and 2.
    assert {c.page_number for c in chunks} == {1, 2}

    # All page-1 chunks come before the page-2 ones.
    page_numbers = [c.page_number for c in chunks]
    assert page_numbers == sorted(page_numbers)


def test_overlap_between_consecutive_blocks():
    extracted = _extracted("a" * 250)
    chunks = chunk_document("doc", extracted, chunk_size=100, overlap=20)

    assert len(chunks) >= 2
    # The next block starts 20 characters BEFORE the end of the previous one.
    assert chunks[1].char_start == chunks[0].char_end - 20


def test_empty_text_produces_no_chunks():
    extracted = _extracted("   ", "")
    chunks = chunk_document("doc", extracted, chunk_size=100, overlap=20)
    assert chunks == []


def test_invalid_overlap_raises():
    extracted = _extracted("text")
    # overlap >= chunk_size would make the window never advance (infinite loop).
    with pytest.raises(ValueError):
        chunk_document("doc", extracted, chunk_size=100, overlap=100)


def test_does_not_split_words_when_possible():
    # Text with spaces: the cut should land on a space, not mid-word.
    text = "word " * 40  # ~200 characters, with regular spaces
    extracted = _extracted(text.strip())
    chunks = chunk_document("doc", extracted, chunk_size=50, overlap=10)

    # No chunk should start or end by splitting "word".
    for c in chunks[:-1]:
        assert not c.text.endswith("wo")
        assert not c.text.endswith("wor")
