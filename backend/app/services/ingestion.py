"""Document ingestion pipeline: bytes in, embedded chunks out.

The notebook did load -> split -> embed -> ``FAISS.from_documents`` in one
synchronous cell. In a service that work has to survive a request boundary, so
it is modelled as an explicit state machine on the ``documents`` row
(``pending -> processing -> ready|failed``) driven by a background task. The
client polls the document until it reaches a terminal state.

Running the pipeline in-process keeps the free-tier deployment to two
containers. The seam for scaling out is :func:`ingest_document`: point a Celery
or ARQ worker at it and the rest of the code is unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Document, DocumentChunk, DocumentStatus, SourceType
from app.providers.base import ModelProvider, ProviderError
from app.services.chunking import split_text
from app.services.extraction import extract_text

logger = logging.getLogger(__name__)


class DuplicateDocumentError(ValueError):
    """The same bytes were already ingested by this user."""

    def __init__(self, existing: Document) -> None:
        super().__init__(f"document already exists: {existing.id}")
        self.existing = existing


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _storage_path(owner_id: uuid.UUID, document_id: uuid.UUID, filename: str | None) -> Path:
    suffix = Path(filename or "").suffix[:10]
    directory = Path(settings.storage_dir) / str(owner_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{document_id}{suffix}"


async def create_document(
    db: AsyncSession,
    *,
    owner_id: uuid.UUID,
    title: str,
    data: bytes,
    mime_type: str,
    source_type: SourceType,
    filename: str | None = None,
    source_uri: str | None = None,
    persist_raw: bool = True,
) -> Document:
    """Register a document in ``pending`` state and store its raw bytes.

    Raises :class:`DuplicateDocumentError` when this owner already holds an
    identical file, which is what the ``(owner_id, checksum)`` unique
    constraint enforces at the database level.
    """
    digest = checksum(data)
    existing = await db.scalar(
        select(Document).where(Document.owner_id == owner_id, Document.checksum == digest)
    )
    if existing is not None:
        raise DuplicateDocumentError(existing)

    document = Document(
        owner_id=owner_id,
        title=title[:500],
        source_type=source_type,
        source_uri=source_uri,
        filename=filename,
        mime_type=mime_type,
        size_bytes=len(data),
        checksum=digest,
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    await db.flush()

    if persist_raw:
        path = _storage_path(owner_id, document.id, filename)
        path.write_bytes(data)
        document.storage_path = str(path)

    await db.commit()
    await db.refresh(document)
    return document


async def ingest_document(
    document_id: uuid.UUID,
    provider: ModelProvider,
    *,
    raw_text: str | None = None,
) -> None:
    """Extract, chunk, embed, and persist. Never raises: failures land on the row.

    Opens its own session because it runs after the HTTP response has been
    sent and the request-scoped session is already closed.
    """
    async with SessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            logger.warning(
                "ingest skipped, document missing",
                extra={"document_id": str(document_id)},
            )
            return

        document.status = DocumentStatus.PROCESSING
        document.error = None
        await db.commit()

        try:
            text = raw_text if raw_text is not None else _read_source(document)
            chunks = split_text(
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not chunks:
                raise ValueError("document produced no text to index")

            # Replace any chunks from an earlier attempt so retries are idempotent.
            await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))

            batch = settings.embed_batch_size
            for start in range(0, len(chunks), batch):
                window = chunks[start : start + batch]
                vectors = await provider.embed_documents([c.text for c in window])
                db.add_all(
                    DocumentChunk(
                        document_id=document.id,
                        owner_id=document.owner_id,
                        chunk_index=chunk.index,
                        content=chunk.text,
                        token_estimate=chunk.token_estimate,
                        embedding=vector,
                        meta={
                            "start_char": chunk.start_char,
                            "title": document.title,
                            "source_uri": document.source_uri,
                        },
                    )
                    for chunk, vector in zip(window, vectors, strict=True)
                )
                await db.flush()

            document.chunk_count = len(chunks)
            document.status = DocumentStatus.READY
            await db.commit()
            logger.info(
                "document ingested",
                extra={"document_id": str(document.id), "chunks": len(chunks)},
            )
        except Exception as exc:
            await db.rollback()
            document = await db.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error = _user_facing_error(exc)
                document.chunk_count = 0
                await db.commit()
            logger.exception("ingestion failed", extra={"document_id": str(document_id)})


def _read_source(document: Document) -> str:
    if not document.storage_path:
        raise ValueError("document has no stored content to index")
    data = Path(document.storage_path).read_bytes()
    return extract_text(data, document.mime_type or "text/plain")


def _user_facing_error(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return f"Embedding provider error: {exc}"
    return f"{type(exc).__name__}: {exc}"[:1000]


async def delete_document(db: AsyncSession, document: Document) -> None:
    """Remove the row (chunks cascade) and the stored file."""
    path = Path(document.storage_path) if document.storage_path else None
    await db.delete(document)
    await db.commit()
    if path is not None:
        path.unlink(missing_ok=True)
