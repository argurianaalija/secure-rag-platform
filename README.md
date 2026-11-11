# Secure RAG Intelligence Platform

A FastAPI backend for a secure Retrieval-Augmented Generation (RAG) system over
PDF documents. It is built incrementally; security is a first-class concern from
day one (hence *Secure* RAG).

**Status:** Weeks 1–2 complete and tested; Week 3 (embeddings + semantic search)
implemented, end-to-end validation in progress.

> Roadmap: upload & API (Week 1), text extraction & chunking (Week 2),
> embeddings + pgvector (Week 3), RAG pipeline (Week 4), AI security (Week 5),
> evaluation (Week 6), production/Docker (Week 7), portfolio (Week 8).

## What it does

- `POST /documents` — upload a PDF. Validates the file and stores it on disk.
- `GET /documents/{document_id}/chunks` — extract the PDF text and split it into
  *chunks* with overlap, keeping page number and offsets.
- `POST /documents/{document_id}/index` — extract, split, compute **embeddings**
  and store the chunks in Postgres/pgvector.
- `GET /search?q=...` — **semantic search**: returns the chunks closest to the
  question (by meaning, not by exact words).
- `GET /health` — health check.
- `GET /docs` — interactive documentation (Swagger UI).

## Architecture

```
Client ──▶ FastAPI ──▶ validation + storage (PDF on disk)
                   ──▶ extraction (PyMuPDF) ──▶ chunking (overlap)
                   ──▶ embeddings (Sentence Transformers)
                   ──▶ Postgres + pgvector (vector search)
```

## Project layout

```
secure-rag-platform/
├── app/
│   ├── main.py            # app factory, routers, error handlers, DB startup
│   ├── config.py          # typed configuration (environment variables)
│   ├── schemas.py         # Pydantic response models
│   ├── errors.py          # domain exceptions + centralized handlers
│   ├── db.py              # Postgres/pgvector: schema, insert, search (Week 3)
│   ├── routers/
│   │   ├── upload.py      # POST /documents, GET .../chunks, POST .../index
│   │   └── search.py      # GET /search (semantic search, Week 3)
│   └── services/
│       ├── storage.py     # PDF validation + storage
│       ├── extraction.py  # text extraction with PyMuPDF (Week 2)
│       ├── chunking.py    # chunking with overlap (Week 2)
│       └── embeddings.py  # Sentence Transformers (Week 3)
├── scripts/
│   └── init_db.py         # create the DB schema manually
├── docker-compose.yml     # Postgres + pgvector (Week 3)
├── tests/
│   ├── test_upload.py     # happy path and error cases
│   ├── test_chunking.py   # unit tests for the chunking logic
│   ├── test_processing.py # end-to-end: from upload to chunks
│   └── test_search.py     # end-to-end: indexing + search (requires DB)
├── uploads/               # uploaded files (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

1. Create a virtual environment and install dependencies:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   **macOS / Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) Copy `.env.example` to `.env` and adjust the values.

3. Start the vector database (needed for indexing/search):
   ```bash
   docker compose up -d
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the docs: <http://127.0.0.1:8000/docs> and try it from there.

### Example with curl

```bash
# upload
curl -F "file=@path/to/document.pdf" http://127.0.0.1:8000/documents
# inspect chunks
curl http://127.0.0.1:8000/documents/<DOCUMENT_ID>/chunks
# index, then search
curl -X POST http://127.0.0.1:8000/documents/<DOCUMENT_ID>/index
curl "http://127.0.0.1:8000/search?q=your+question"
```

## Tests

```bash
pytest
```

Unit and integration tests cover upload validation and the chunking logic. The
semantic-search test (`test_search.py`) is automatically skipped when the
embedding backend or the database is unavailable, so the suite stays green.

## Security decisions (from Week 1)

| Risk | Mitigation |
|---|---|
| Forged extension / Content-Type | Check the real **magic bytes** (`%PDF-`) + MIME via `python-magic` |
| Huge file exhausting RAM (DoS) | **Streaming** read that stops as soon as the limit is exceeded |
| Path traversal via file name | **Generated** on-disk name (UUID), never derived from user input |
| Exposed stack traces | Generic `500` handler that leaks no internal details |
| Inconsistent errors | **Single** error format (`{error, detail}`) across the API |

## Week 2 — Document processing

Given an uploaded PDF, we extract its text (PyMuPDF), split it into overlapping
*chunks* and keep the page and offsets.

Design notes:
- **per-page chunks**: each block belongs to exactly one page, so citations
  ("page 7") will be accurate;
- **overlap** (`chunk_overlap_chars`): a sentence sitting across two blocks
  stays intact in at least one of them and is not lost during retrieval;
- **cut on spaces**: we avoid splitting words in half.

## Week 3 — Embeddings and semantic search

Flow: upload a PDF, **index** it (`POST .../index`), then run **semantic
search** (`GET /search`).

Key concepts:
- **embedding**: a text becomes a 384-number vector capturing its meaning
  (`all-MiniLM-L6-v2` model);
- **pgvector**: Postgres stores the vectors and compares them with the `<=>`
  operator (cosine distance); `1 - distance` = similarity;
- search finds chunks **by meaning**, even when they do not contain the same
  words as the question.

## Next step (Week 4)

Connect retrieval to an LLM: build the system prompt, generate answers grounded
in the retrieved chunks, add citations (page number) and handle unanswerable
questions.
