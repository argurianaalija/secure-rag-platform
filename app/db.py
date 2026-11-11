"""Access to the Postgres + pgvector vector database (Week 3).

Everything related to the DB lives here: schema creation, insertion of chunks
with their embeddings, and similarity search.

Key pgvector concepts:
- the `vector(N)` type stores a vector of N numbers (N = 384 for us);
- the `<=>` operator computes the **cosine distance** between two vectors;
- `1 - (a <=> b)` is therefore the **cosine similarity** (1 = identical, 0 = orthogonal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.config import Settings


@dataclass
class SearchHit:
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float  # cosine similarity (higher = more relevant)


def _connect(settings: Settings) -> psycopg.Connection:
    """Open a connection and register the `vector` type on it.

    `register_vector` needs the `vector` extension to already exist in the DB
    (we create it in `init_schema`).
    """
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema(settings: Settings) -> None:
    """Create the extension, the chunks table and the similarity index.

    Idempotent: it can be called any number of times (IF NOT EXISTS).
    """
    conn = psycopg.connect(settings.database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           BIGSERIAL PRIMARY KEY,
                    document_id  TEXT NOT NULL,
                    chunk_index  INT  NOT NULL,
                    page_number  INT  NOT NULL,
                    text         TEXT NOT NULL,
                    embedding    vector({settings.embedding_dim}) NOT NULL,
                    UNIQUE (document_id, chunk_index)
                )
                """
            )
            # HNSW index for fast (approximate) similarity searches.
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        conn.commit()
    finally:
        conn.close()


def replace_document_chunks(
    settings: Settings,
    document_id: str,
    rows: list[tuple[int, int, str, np.ndarray]],
) -> int:
    """Replace a document's chunks (delete first, then insert).

    `rows` = list of (chunk_index, page_number, text, embedding).
    Deleting first makes the operation repeatable without creating duplicates
    (idempotency): re-indexing the same document always yields the same state.
    """
    conn = _connect(settings)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, page_number, text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [(document_id, idx, page, text, emb) for idx, page, text, emb in rows],
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def search_similar(
    settings: Settings,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[SearchHit]:
    """Return the `top_k` chunks most similar to the question's vector."""
    conn = _connect(settings)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, chunk_index, page_number, text,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            return [
                SearchHit(
                    document_id=r[0],
                    chunk_index=r[1],
                    page_number=r[2],
                    text=r[3],
                    score=float(r[4]),
                )
                for r in cur.fetchall()
            ]
    finally:
        conn.close()
