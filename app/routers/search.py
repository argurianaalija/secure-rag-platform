"""Semantic search endpoint (Week 3).

Given a natural-language question, we compute its embedding and look in the DB
for the chunks whose vector is closest (cosine similarity). This is the
"retrieval" step that, in Week 4, will feed the LLM.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import Settings, get_settings
from app.schemas import SearchHitOut, SearchResponse
from app.services.embeddings import embed_query
from app import db

router = APIRouter(tags=["search"])


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over the indexed chunks",
)
async def search(
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(min_length=1, description="The question / text to search for.")],
    top_k: Annotated[int | None, Query(ge=1, le=50, description="How many results.")] = None,
) -> SearchResponse:
    k = top_k or settings.search_top_k
    query_vec = embed_query(q, settings)
    hits = db.search_similar(settings, query_vec, k)

    return SearchResponse(
        query=q,
        results=[
            SearchHitOut(
                document_id=h.document_id,
                chunk_index=h.chunk_index,
                page_number=h.page_number,
                text=h.text,
                score=h.score,
            )
            for h in hits
        ],
    )
