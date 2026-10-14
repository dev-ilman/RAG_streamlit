from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.chat import MessageRole


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=300)


class ChatSessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[dict[str, Any]]
    model: str | None
    latency_ms: int | None
    created_at: datetime


class ChatSessionDetail(ChatSessionRead):
    messages: list[ChatMessageRead]


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = None


class AskResponse(BaseModel):
    session_id: uuid.UUID
    user_message: ChatMessageRead
    assistant_message: ChatMessageRead
