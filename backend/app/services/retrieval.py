"""Vector search over ``document_chunks``.

This is where FAISS is replaced. Instead of an in-process index rebuilt on
every start, similarity search runs inside Postgres via pgvector, so it is
durable, shared across API replicas, transactional with the rest of the data,
and - importantly - filterable: ownership and document scoping are applied by
the same query that does the ANN lookup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.providers.base import ModelProvider


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    score: float

    def as_citation(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "document_title": self.document_title,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "excerpt": self.content[:400],
        }


def _base_query(
    owner_id: uuid.UUID, document_ids: list[uuid.UUID] | None
) -> Select[tuple[DocumentChunk, str, float]]:
    query = select(DocumentChunk, Document.title).join(
        Document, Document.id == DocumentChunk.document_id
    )
    query = query.where(
        DocumentChunk.owner_id == owner_id,
        Document.status == DocumentStatus.READY,
    )
    if document_ids:
        query = query.where(DocumentChunk.document_id.in_(document_ids))
    return query  # type: ignore[return-value]


async def search(
    db: AsyncSession,
    *,
    owner_id: uuid.UUID,
    query_text: str,
    provider: ModelProvider,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` most similar chunks the user is allowed to see."""
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score

    query_vector = await provider.embed_query(query_text)

    # pgvector's `<=>` is cosine *distance* in [0, 2]; similarity is 1 - distance.
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    stmt = (
        _base_query(owner_id, document_ids)
        .add_columns(distance.label("distance"))
        .order_by(distance)
        .limit(top_k)
    )

    rows = (await db.execute(stmt)).all()
    results = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=1.0 - float(dist),
        )
        for chunk, title, dist in rows
    ]
    return [r for r in results if r.score >= min_score]
