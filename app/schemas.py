"""Pydantic models for the API responses.

Defining explicit schemas serves two purposes:
1. correct automatic documentation (OpenAPI / Swagger);
2. a stable contract towards the frontend / other services.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after a successful upload."""

    document_id: str = Field(..., description="Unique identifier of the stored document.")
    filename: str = Field(..., description="Original name of the uploaded file.")
    size_bytes: int = Field(..., description="File size in bytes.")
    content_type: str = Field(..., description="Verified MIME type of the file.")
    message: str = Field(default="Upload completed successfully.")


class ErrorResponse(BaseModel):
    """Standard shape of every error returned by the API."""

    error: str = Field(..., description="Machine-readable error code, e.g. 'file_too_large'.")
    detail: str = Field(..., description="Human-readable message.")


class ChunkOut(BaseModel):
    """A single text block extracted from the document."""

    index: int = Field(..., description="Position of the chunk in the document (0, 1, 2, ...).")
    text: str = Field(..., description="The text of the block.")
    page_number: int = Field(..., description="PDF page it comes from (1-based).")
    char_start: int = Field(..., description="Start offset of the block within the page.")
    char_end: int = Field(..., description="End offset of the block within the page.")


class ChunkingResponse(BaseModel):
    """Result of splitting a document into chunks."""

    document_id: str
    page_count: int = Field(..., description="Number of pages in the PDF.")
    chunk_count: int = Field(..., description="Total number of chunks produced.")
    chunk_size_chars: int
    chunk_overlap_chars: int
    chunks: list[ChunkOut]


class IndexResponse(BaseModel):
    """Outcome of indexing a document (chunks + embeddings stored)."""

    document_id: str
    chunks_indexed: int = Field(..., description="Number of chunks saved in the DB.")
    embedding_model: str
    message: str = "Document indexed successfully."


class SearchHitOut(BaseModel):
    """A single semantic-search result."""

    document_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float = Field(..., description="Cosine similarity (1 = most relevant).")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHitOut]


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str
