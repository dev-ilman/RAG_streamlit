"""Chat sessions, grounded answers, and a streaming variant."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.deps import CurrentUser, DbSession, Provider
from app.db.session import SessionLocal
from app.models import ChatMessage, ChatSession, MessageRole
from app.providers.base import ModelProvider, ProviderError
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionRead,
    ChatSessionUpdate,
)
from app.services import rag, retrieval

router = APIRouter(prefix="/chat", tags=["chat"])


async def _get_session(db: DbSession, session_id: uuid.UUID, owner_id: uuid.UUID) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None or session.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")
    return session


async def _recent_history(
    db: DbSession, session_id: uuid.UUID, limit: int
) -> list[tuple[str, str]]:
    """The last ``limit`` turns, oldest first, for multi-turn context."""
    rows = (
        await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
    ).all()
    return [(m.role.value, m.content) for m in reversed(rows)]


def _derive_title(text: str) -> str:
    condensed = " ".join(text.split())
    return (condensed[:57] + "...") if len(condensed) > 60 else condensed or "New chat"


@router.get("/sessions", response_model=list[ChatSessionRead])
async def list_sessions(user: CurrentUser, db: DbSession, limit: int = 50) -> list[ChatSessionRead]:
    rows = await db.scalars(
        select(ChatSession)
        .where(ChatSession.owner_id == user.id)
        .order_by(ChatSession.last_message_at.desc())
        .limit(max(1, min(limit, 200)))
    )
    return [ChatSessionRead.model_validate(s) for s in rows]


@router.post("/sessions", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: ChatSessionCreate, user: CurrentUser, db: DbSession
) -> ChatSessionRead:
    session = ChatSession(owner_id=user.id, title=payload.title or "New chat")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return ChatSessionRead.model_validate(session)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> ChatSessionDetail:
    session = await db.scalar(
        select(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.owner_id == user.id)
        .options(selectinload(ChatSession.messages))
    )
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")
    return ChatSessionDetail.model_validate(session)


@router.patch("/sessions/{session_id}", response_model=ChatSessionRead)
async def rename_session(
    session_id: uuid.UUID, payload: ChatSessionUpdate, user: CurrentUser, db: DbSession
) -> ChatSessionRead:
    session = await _get_session(db, session_id, user.id)
    session.title = payload.title
    await db.commit()
    await db.refresh(session)
    return ChatSessionRead.model_validate(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    await db.delete(await _get_session(db, session_id, user.id))
    await db.commit()


@router.post("/sessions/{session_id}/ask", response_model=AskResponse)
async def ask(
    session_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
) -> AskResponse:
    """Retrieve, generate, and persist both turns in one transaction.

    Storing the citations alongside the answer means an answer can always be
    traced back to the exact passages that produced it, even after the
    underlying documents change.
    """
    session = await _get_session(db, session_id, user.id)
    question = payload.message.strip()

    history = await _recent_history(db, session.id, settings.max_history_messages)
    chunks = await retrieval.search(
        db,
        owner_id=user.id,
        query_text=question,
        provider=provider,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )

    if chunks:
        result, latency_ms = await rag.answer(
            provider=provider, question=question, chunks=chunks, history=history
        )
        answer_text, model_name = result.text, result.model
    else:
        answer_text, model_name, latency_ms = rag.NO_CONTEXT_ANSWER, None, 0

    user_message = ChatMessage(
        session_id=session.id, role=MessageRole.USER, content=question, citations=[]
    )
    assistant_message = ChatMessage(
        session_id=session.id,
        role=MessageRole.ASSISTANT,
        content=answer_text,
        citations=[c.as_citation() for c in chunks],
        model=model_name,
        latency_ms=latency_ms,
        prompt_chars=sum(len(c.content) for c in chunks),
    )
    db.add_all([user_message, assistant_message])

    if not history:
        session.title = _derive_title(question)
    session.last_message_at = func.now()
    await db.commit()
    await db.refresh(user_message)
    await db.refresh(assistant_message)

    return AskResponse(
        session_id=session.id,
        user_message=ChatMessageRead.model_validate(user_message),
        assistant_message=ChatMessageRead.model_validate(assistant_message),
    )


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_and_persist(
    *,
    session_id: uuid.UUID,
    question: str,
    chunks: list[retrieval.RetrievedChunk],
    history: list[tuple[str, str]],
    provider: ModelProvider,
    is_first_turn: bool,
) -> AsyncIterator[str]:
    """Emit SSE frames, then write both turns with a fresh session.

    The request-scoped session is already unwound by the time this generator
    runs, so persistence opens its own.
    """
    citations = [c.as_citation() for c in chunks]
    yield _sse("meta", {"session_id": str(session_id), "citations": citations})

    started = time.perf_counter()
    parts: list[str] = []
    try:
        if chunks:
            async for delta in rag.stream_answer(
                provider=provider, question=question, chunks=chunks, history=history
            ):
                parts.append(delta)
                yield _sse("token", {"text": delta})
        else:
            parts.append(rag.NO_CONTEXT_ANSWER)
            yield _sse("token", {"text": rag.NO_CONTEXT_ANSWER})
    except ProviderError as exc:
        yield _sse("error", {"message": str(exc)})
        return

    answer_text = "".join(parts).strip()
    latency_ms = int((time.perf_counter() - started) * 1000)

    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        if session is None:  # deleted mid-stream
            yield _sse("error", {"message": "Chat session no longer exists"})
            return
        assistant_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            citations=citations,
            model=provider.chat_model,
            latency_ms=latency_ms,
            prompt_chars=sum(len(c.content) for c in chunks),
        )
        db.add_all(
            [
                ChatMessage(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=question,
                    citations=[],
                ),
                assistant_message,
            ]
        )
        if is_first_turn:
            session.title = _derive_title(question)
        session.last_message_at = func.now()
        await db.commit()
        message_id = str(assistant_message.id)

    yield _sse("done", {"message_id": message_id, "latency_ms": latency_ms})


@router.post("/sessions/{session_id}/ask/stream")
async def ask_stream(
    session_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUser,
    db: DbSession,
    provider: Provider,
) -> StreamingResponse:
    """Server-sent events: `meta` (citations), many `token`, then `done`."""
    session = await _get_session(db, session_id, user.id)
    question = payload.message.strip()
    history = await _recent_history(db, session.id, settings.max_history_messages)
    chunks = await retrieval.search(
        db,
        owner_id=user.id,
        query_text=question,
        provider=provider,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )

    # Every read this handler needs is done. Release the connection now rather
    # than holding it open for the whole answer: a streamed reply can run for
    # tens of seconds, and the pool is far smaller than the number of readers.
    await db.close()

    return StreamingResponse(
        _stream_and_persist(
            session_id=session.id,
            question=question,
            chunks=chunks,
            history=history,
            provider=provider,
            is_first_turn=not history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # stop nginx from buffering the stream
            "Connection": "keep-alive",
        },
    )
