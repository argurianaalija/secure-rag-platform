"""Splitting text into blocks (chunks) with overlap (Week 2).

Why do we need chunks?
An LLM and a semantic search engine work better on small, coherent pieces of
text, not on a whole document. So, when the user asks a question, we retrieve
only the 3-4 most relevant pieces instead of the entire PDF.

Why the overlap?
If we cut the text into clean blocks, an important sentence could end up split
between two blocks and no longer be retrievable as a whole. By making the end of
one block repeat at the start of the next, the idea stays intact in at least one.

Chosen strategy (simple and robust to start):
- we split **page by page**, so each chunk certainly belongs to one specific
  page (essential for citations);
- within each page we use a sliding window of ~`chunk_size` characters with
  `overlap` characters of overlap, trying to cut on a space so we don't split
  words.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.extraction import ExtractedDocument


@dataclass
class Chunk:
    document_id: str
    index: int  # position of the chunk in the document (0, 1, 2, ...)
    text: str
    page_number: int
    char_start: int  # start offset within the page
    char_end: int  # end offset within the page


def _split_page(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split the text of ONE page into (text, start, end).

    Sliding window with overlap; tries to cut on a space so it does not split
    words in half.
    """
    text = text.strip()
    if not text:
        return []

    pieces: list[tuple[str, int, int]] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # If we are not at the end of the text, move back to the last space:
        # this avoids cutting a word. If there is no space in the window, cut hard.
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space

        piece = text[start:end].strip()
        if piece:
            pieces.append((piece, start, end))

        if end >= n:
            break

        # The next block starts a bit earlier, to create the overlap.
        start = max(end - overlap, start + 1)

    return pieces


def chunk_document(
    document_id: str,
    extracted: ExtractedDocument,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Turn an extracted document into a list of numbered Chunks.

    Defensive validation: overlap must be smaller than chunk_size, otherwise the
    window would never advance (infinite loop).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[Chunk] = []
    index = 0
    for page in extracted.pages:
        for piece, c_start, c_end in _split_page(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    index=index,
                    text=piece,
                    page_number=page.page_number,
                    char_start=c_start,
                    char_end=c_end,
                )
            )
            index += 1

    return chunks
