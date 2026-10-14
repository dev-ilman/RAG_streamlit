"""Prompt assembly and answer generation.

The system prompt keeps the notebook's grounding contract - answer only from
the supplied context - and adds the two things a user-facing product needs:
numbered sources the model is told to cite, and an explicit instruction to
admit when the context does not contain the answer.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from app.providers.base import ChatTurn, CompletionResult, ModelProvider
from app.services.retrieval import RetrievedChunk

SYSTEM_PROMPT_TEMPLATE = """\
You are a retrieval-augmented assistant. Answer the question using only the \
context below. Think through the context before answering, and be accurate and \
specific.

Rules:
- Use only facts present in the context. Do not rely on outside knowledge.
- Cite the sources you used inline as [1], [2], matching the numbering below.
- If the context does not contain the answer, say so plainly and state what \
additional information would be needed. Do not guess.
- Keep the answer focused; prefer specifics from the context over generalities.

<context>
{context}
</context>"""

NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in your documents to answer that. "
    "Upload a document that covers the topic, or rephrase the question."
)


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, citable block."""
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{position}] source: {chunk.document_title} "
            f"(chunk {chunk.chunk_index})\n{chunk.content}"
        )
    return "\n\n".join(blocks)


def build_system_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context=build_context(chunks))


def to_turns(history: Sequence[tuple[str, str]]) -> list[ChatTurn]:
    return [ChatTurn(role=role, content=content) for role, content in history]


async def answer(
    *,
    provider: ModelProvider,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[tuple[str, str]] = (),
) -> tuple[CompletionResult, int]:
    """Generate an answer. Returns ``(result, latency_ms)``."""
    started = time.perf_counter()
    system_prompt = build_system_prompt(chunks)
    result = await provider.complete(
        system_prompt=system_prompt,
        history=to_turns(history),
        user_message=question,
    )
    return result, int((time.perf_counter() - started) * 1000)


def stream_answer(
    *,
    provider: ModelProvider,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[tuple[str, str]] = (),
) -> AsyncIterator[str]:
    return provider.stream(
        system_prompt=build_system_prompt(chunks),
        history=to_turns(history),
        user_message=question,
    )
