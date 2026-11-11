"""Validation and storage of uploaded PDF files.

Security choices (important from Week 1, because this is a "Secure RAG" project):

1. We trust neither the file extension nor the Content-Type declared by the
   client: both are trivially forgeable. We check the real *magic bytes*.
2. We read the file in blocks (streaming) and stop as soon as we exceed the
   size limit, instead of loading everything into memory: this way a huge file
   cannot exhaust the server's RAM (a form of DoS).
3. We generate the stored file name ourselves (UUID) and never use the
   user-provided name to build the path on disk: this prevents path traversal
   such as "../../etc/passwd".
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.errors import (
    CorruptedFileError,
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    InvalidDocumentIdError,
    InvalidFileTypeError,
)

# A valid document_id is exactly the hex form of a UUID (32 characters).
# Validating it blocks path traversal like "../../something" at the source.
_DOCUMENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# PDFs always start with these bytes: "%PDF-"
PDF_MAGIC = b"%PDF-"
CHUNK_SIZE = 1024 * 1024  # read 1 MB at a time


@dataclass
class StoredDocument:
    document_id: str
    original_filename: str
    size_bytes: int
    content_type: str
    path: Path


def _detect_mime(sample: bytes) -> str:
    """Detect the real MIME type from the first bytes of the file.

    Uses python-magic if available; otherwise falls back to the PDF magic bytes.
    """
    try:
        import magic  # local import: the library may not be installed

        return magic.from_buffer(sample, mime=True)
    except Exception:
        # Minimal but sufficient fallback for PDFs.
        return "application/pdf" if sample.startswith(PDF_MAGIC) else "application/octet-stream"


async def validate_and_store(file: UploadFile, settings: Settings) -> StoredDocument:
    """Validate an UploadFile and, if it is a valid PDF, store it on disk.

    Raises domain exceptions (AppError) on problems; the centralized handler
    turns them into consistent HTTP responses.
    """
    # 1) There must be a file name.
    if not file.filename:
        raise InvalidFileTypeError("No file received.")

    # 2) Read the first block to check the magic bytes BEFORE storing anything.
    first_chunk = await file.read(CHUNK_SIZE)
    if not first_chunk:
        raise EmptyFileError("The file is empty.")

    if not first_chunk.startswith(PDF_MAGIC):
        raise CorruptedFileError(
            "The file content is not a valid PDF (missing header)."
        )

    detected_mime = _detect_mime(first_chunk)
    if detected_mime not in settings.allowed_content_types:
        raise InvalidFileTypeError(
            f"File type not allowed: '{detected_mime}'. Only PDFs are accepted."
        )

    # 3) Write to disk in a streaming fashion, counting bytes and stopping if we
    #    exceed the limit. The stored file has a name we generate ourselves.
    document_id = uuid.uuid4().hex
    dest_path = settings.upload_path / f"{document_id}.pdf"
    total = 0

    try:
        with dest_path.open("wb") as out:
            chunk = first_chunk
            while chunk:
                total += len(chunk)
                if total > settings.max_upload_size_bytes:
                    out.close()
                    dest_path.unlink(missing_ok=True)  # do not leave partial files
                    raise FileTooLargeError(
                        f"File too large. Limit: {settings.max_upload_size_mb} MB."
                    )
                out.write(chunk)
                chunk = await file.read(CHUNK_SIZE)
    except FileTooLargeError:
        raise
    except OSError as exc:  # disk write error
        dest_path.unlink(missing_ok=True)
        raise CorruptedFileError(f"Unable to store the file: {exc}") from exc

    return StoredDocument(
        document_id=document_id,
        original_filename=file.filename,
        size_bytes=total,
        content_type=detected_mime,
        path=dest_path,
    )


def resolve_document_path(document_id: str, settings: Settings) -> Path:
    """Return the path of the stored PDF for a given document_id.

    - Validates the id format (no path traversal).
    - Checks that the file actually exists, otherwise 404.
    """
    if not _DOCUMENT_ID_RE.match(document_id):
        raise InvalidDocumentIdError("Invalid document_id format.")

    path = settings.upload_path / f"{document_id}.pdf"
    if not path.is_file():
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")

    return path
