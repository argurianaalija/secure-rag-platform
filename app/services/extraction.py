"""Extracting text from a PDF, page by page (Week 2).

We use PyMuPDF (imported as `pymupdf`, historically also known as `fitz`).
We extract the text *keeping the page number*: we will need it later for
citations ("this answer comes from page 7").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # the modern name is `pymupdf`; older versions use `fitz`
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore

from app.errors import CorruptedFileError


@dataclass
class PageText:
    """The text of a single page."""

    page_number: int  # 1-based: the first page is 1, not 0
    text: str


@dataclass
class ExtractedDocument:
    """Extraction result: all pages + the PDF metadata."""

    pages: list[PageText]
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def extract_text(pdf_path: Path) -> ExtractedDocument:
    """Open a PDF and extract the text of each page.

    Raises CorruptedFileError if the file cannot be opened as a PDF.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # unreadable file / not a real PDF
        raise CorruptedFileError(f"Unable to open the PDF: {exc}") from exc

    try:
        pages = [
            # get_text("text") returns the "flat" text of the page,
            # in the reading order reconstructed by PyMuPDF.
            PageText(page_number=i, text=page.get_text("text"))
            for i, page in enumerate(doc, start=1)
        ]
        # doc.metadata is a dict with author, title, etc. (may be empty).
        metadata = dict(doc.metadata or {})
    finally:
        doc.close()  # always close the file, even on error

    return ExtractedDocument(pages=pages, metadata=metadata)
