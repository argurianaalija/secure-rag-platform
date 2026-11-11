"""PDF upload and processing endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import (
    ChunkingResponse,
    ChunkOut,
    ErrorResponse,
    IndexResponse,
    UploadResponse,
)
from app.services.chunking import chunk_document
from app.services.embeddings import embed_texts
from app.services.extraction import extract_text
from app.services.storage import resolve_document_path, validate_and_store
from app import db

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Empty or corrupted file"},
    },
)
async def upload_document(
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="The PDF file to upload.")],
) -> UploadResponse:
    """Receive a PDF, validate it and store it.

    Validation and storage live in the `storage` service; here the router only
    orchestrates and builds the response. On problems, the service raises domain
    exceptions handled centrally in `errors.py`.
    """
    stored = await validate_and_store(file, settings)
    return UploadResponse(
        document_id=stored.document_id,
        filename=stored.original_filename,
        size_bytes=stored.size_bytes,
        content_type=stored.content_type,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkingResponse,
    summary="Extract a document's text and split it into chunks",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid document_id"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "Corrupted or unreadable PDF"},
    },
)
async def get_document_chunks(
    document_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChunkingResponse:
    """Given an already-uploaded document, extract the text and split it into chunks.

    For now the computation is done "on the fly" (we do not store the chunks):
    in Week 3 we store them in the database together with their embeddings.
    """
    path = resolve_document_path(document_id, settings)
    extracted = extract_text(path)
    chunks = chunk_document(
        document_id=document_id,
        extracted=extracted,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )

    return ChunkingResponse(
        document_id=document_id,
        page_count=extracted.page_count,
        chunk_count=len(chunks),
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        chunks=[
            ChunkOut(
                index=c.index,
                text=c.text,
                page_number=c.page_number,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for c in chunks
        ],
    )


@router.post(
    "/{document_id}/index",
    response_model=IndexResponse,
    summary="Extract, split, compute embeddings and store in the DB",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid document_id"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "Corrupted or unreadable PDF"},
    },
)
async def index_document(
    document_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> IndexResponse:
    """Full ingestion pipeline for an already-uploaded document:

    text extraction -> chunking -> embeddings -> store in Postgres/pgvector.
    Re-indexing the same document replaces the previous chunks (idempotent).
    """
    path = resolve_document_path(document_id, settings)
    extracted = extract_text(path)
    chunks = chunk_document(
        document_id=document_id,
        extracted=extracted,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )

    # A single encode call for all chunks: much more efficient than one at a time.
    vectors = embed_texts([c.text for c in chunks], settings) if chunks else []
    rows = [
        (c.index, c.page_number, c.text, vec)
        for c, vec in zip(chunks, vectors)
    ]
    count = db.replace_document_chunks(settings, document_id, rows)

    return IndexResponse(
        document_id=document_id,
        chunks_indexed=count,
        embedding_model=settings.embedding_model_name,
    )
