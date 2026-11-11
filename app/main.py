"""Application entry point for the FastAPI app.

Here we wire the pieces together:
- create the app,
- on startup, prepare the database schema (table + pgvector extension),
- register the centralized error handlers,
- mount the routers,
- expose a health-check endpoint.

Run locally (start the DB first: `docker compose up -d`):
    uvicorn app.main:app --reload

Interactive docs once running:
    http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.config import get_settings
from app.errors import register_exception_handlers
from app.routers import search, upload
from app.schemas import HealthResponse

logger = logging.getLogger("secure_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code run at startup and shutdown.

    On startup we try to create the DB schema. If the database is unreachable
    (e.g. you forgot `docker compose up -d`), we do not crash the app: we log a
    warning. Endpoints that use the DB will fail cleanly until the DB is up.
    """
    settings = get_settings()
    try:
        db.init_schema(settings)
        logger.info("Database schema ready.")
    except Exception as exc:  # DB unreachable at startup
        logger.warning("Database unreachable at startup: %s", exc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Secure RAG Intelligence Platform",
        description=(
            "API for PDF upload, chunking, embeddings and semantic search "
            "(Weeks 1-3)."
        ),
        version="0.3.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    app.include_router(upload.router)
    app.include_router(search.router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(environment=settings.environment)

    return app


app = create_app()
