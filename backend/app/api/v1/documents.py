"""Document CRUD, ingestion triggers, and raw vector search."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.core.config import settings
from app.core.deps import CurrentUser, DbSession, Provider
from app.models import Document, DocumentStatus, SourceType
from app.schemas.document import (
    DocumentList,
    DocumentRead,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TextIngestRequest,
    UrlIngestRequest,
)
from app.services import ingestion, retrieval
from app.services.extraction import (
    ExtractionError,
    UnsupportedDocumentError,
    fetch_url,
    guess_mime_type,
)

router = APIRouter(prefix="/documents", tags=["documents"])


async def _get_owned(db: DbSession, document_id: uuid.UUID, owner_id: uuid.UUID) -> Document:
    document = await db.get(Document, document_id)
    # 404 rather than 403 on another user's row: never confirm it exists.
    if document is None or document.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.get("", response_model=DocumentList)
async def list_documents(
    user: CurrentUser,
    db: DbSession,
    limit: int = 50,
    offset: int = 0,
    document_status: DocumentStatus | None = None,
) -> DocumentList:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    filters = [Document.owner_id == user.id]
    if document_status is not None:
        filters.append(Document.status == document_status)

    total = await db.scalar(select(func.count(Document.id)).where(*filters)) or 0
    rows = await db.scalars(
        select(Document)
        .where(*filters)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return DocumentList(
        items=[DocumentRead.model_validate(d) for d in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> DocumentRead:
    """Accept a file, register it, and embed it in the background.

    Returns 202: the row exists immediately, but the document is not
    searchable until its status reaches ``ready``.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {settings.max_upload_bytes // 1024 // 1024} MB limit",
        )
    try:
        mime_type = guess_mime_type(file.filename, file.content_type)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    try:
        document = await ingestion.create_document(
            db,
            owner_id=user.id,
            title=title or file.filename or "Untitled",
            data=data,
            mime_type=mime_type,
            source_type=SourceType.UPLOAD,
            filename=file.filename,
        )
    except ingestion.DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This file was already uploaded (document {exc.existing.id})",
        ) from exc

    background.add_task(ingestion.ingest_document, document.id, provider)
    return DocumentRead.model_validate(document)


@router.post("/url", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def ingest_from_url(
    payload: UrlIngestRequest,
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
    background: BackgroundTasks,
) -> DocumentRead:
    url = str(payload.url)
    try:
        text, _content_type = await fetch_url(url, max_bytes=settings.max_upload_bytes)
    except (ExtractionError, UnsupportedDocumentError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not text.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No readable text at that URL")

    try:
        document = await ingestion.create_document(
            db,
            owner_id=user.id,
            title=payload.title or url,
            data=text.encode("utf-8"),
            mime_type="text/plain",
            source_type=SourceType.URL,
            source_uri=url,
        )
    except ingestion.DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"That page was already ingested (document {exc.existing.id})",
        ) from exc

    background.add_task(ingestion.ingest_document, document.id, provider, raw_text=text)
    return DocumentRead.model_validate(document)


@router.post("/text", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def ingest_text(
    payload: TextIngestRequest,
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
    background: BackgroundTasks,
) -> DocumentRead:
    try:
        document = await ingestion.create_document(
            db,
            owner_id=user.id,
            title=payload.title,
            data=payload.content.encode("utf-8"),
            mime_type="text/plain",
            source_type=SourceType.TEXT,
        )
    except ingestion.DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Identical text already ingested (document {exc.existing.id})",
        ) from exc

    background.add_task(ingestion.ingest_document, document.id, provider, raw_text=payload.content)
    return DocumentRead.model_validate(document)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> DocumentRead:
    return DocumentRead.model_validate(await _get_owned(db, document_id, user.id))


@router.post(
    "/{document_id}/reingest",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reingest_document(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
    background: BackgroundTasks,
) -> DocumentRead:
    """Re-run the pipeline, e.g. after a transient provider failure."""
    document = await _get_owned(db, document_id, user.id)
    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is already being processed")
    document.status = DocumentStatus.PENDING
    document.error = None
    await db.commit()
    await db.refresh(document)
    background.add_task(ingestion.ingest_document, document.id, provider)
    return DocumentRead.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    await ingestion.delete_document(db, await _get_owned(db, document_id, user.id))


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    payload: SearchRequest, user: CurrentUser, db: DbSession, provider: Provider
) -> SearchResponse:
    """Retrieval without generation. Useful for inspecting relevance directly."""
    hits = await retrieval.search(
        db,
        owner_id=user.id,
        query_text=payload.query,
        provider=provider,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
    return SearchResponse(
        query=payload.query,
        hits=[
            SearchHit(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                document_title=h.document_title,
                chunk_index=h.chunk_index,
                score=round(h.score, 4),
                content=h.content,
            )
            for h in hits
        ],
    )
