"""Retrieval behaviour that only shows up against a real pgvector index."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models import Document, DocumentChunk, DocumentStatus, SourceType, User
from app.providers.fake import FakeProvider
from app.services import retrieval


async def _seed_user(db: AsyncSession, email: str) -> User:
    user = User(email=email, hashed_password=hash_password("correct-horse-battery"))
    db.add(user)
    await db.flush()
    return user


async def _seed_document(
    db: AsyncSession,
    user: User,
    provider: FakeProvider,
    title: str,
    passages: list[str],
    status: DocumentStatus = DocumentStatus.READY,
) -> Document:
    document = Document(
        owner_id=user.id,
        title=title,
        source_type=SourceType.TEXT,
        checksum=uuid.uuid4().hex,
        status=status,
        chunk_count=len(passages),
    )
    db.add(document)
    await db.flush()
    vectors = await provider.embed_documents(passages)
    db.add_all(
        DocumentChunk(
            document_id=document.id,
            owner_id=user.id,
            chunk_index=i,
            content=text,
            token_estimate=len(text) // 4,
            embedding=vector,
            meta={},
        )
        for i, (text, vector) in enumerate(zip(passages, vectors, strict=True))
    )
    await db.commit()
    return document


async def test_search_orders_by_similarity(db: AsyncSession, provider: FakeProvider) -> None:
    user = await _seed_user(db, "ranker@example.com")
    await _seed_document(
        db,
        user,
        provider,
        "Mixed corpus",
        [
            "lithium plating happens at low temperature and high charge rate",
            "the pasta water should be salted like the sea",
            "solid electrolyte interphase growth consumes cyclable lithium",
        ],
    )

    hits = await retrieval.search(
        db, owner_id=user.id, query_text="lithium plating low temperature", provider=provider
    )
    assert hits
    assert "lithium plating" in hits[0].content
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


async def test_search_excludes_documents_that_are_not_ready(
    db: AsyncSession, provider: FakeProvider
) -> None:
    user = await _seed_user(db, "pending@example.com")
    await _seed_document(
        db, user, provider, "Half-ingested", ["lithium plating"], DocumentStatus.PROCESSING
    )
    hits = await retrieval.search(
        db, owner_id=user.id, query_text="lithium plating", provider=provider
    )
    assert hits == []


async def test_search_is_scoped_to_the_owner(db: AsyncSession, provider: FakeProvider) -> None:
    owner = await _seed_user(db, "owner@example.com")
    intruder = await _seed_user(db, "intruder@example.com")
    await _seed_document(db, owner, provider, "Private", ["lithium plating happens"])

    assert (
        await retrieval.search(
            db, owner_id=intruder.id, query_text="lithium plating", provider=provider
        )
        == []
    )


async def test_document_filter_narrows_the_corpus(db: AsyncSession, provider: FakeProvider) -> None:
    user = await _seed_user(db, "filter@example.com")
    batteries = await _seed_document(db, user, provider, "Batteries", ["lithium plating"])
    await _seed_document(db, user, provider, "Cooking", ["salt the pasta water"])

    hits = await retrieval.search(
        db,
        owner_id=user.id,
        query_text="salt the pasta water",
        provider=provider,
        document_ids=[batteries.id],
    )
    assert all(h.document_id == batteries.id for h in hits)


async def test_top_k_is_respected(db: AsyncSession, provider: FakeProvider) -> None:
    user = await _seed_user(db, "topk@example.com")
    await _seed_document(
        db, user, provider, "Many", [f"passage number {i} about lithium" for i in range(12)]
    )
    hits = await retrieval.search(
        db, owner_id=user.id, query_text="lithium", provider=provider, top_k=4
    )
    assert len(hits) == 4


async def test_deleting_a_document_cascades_to_its_chunks(
    db: AsyncSession, provider: FakeProvider
) -> None:
    from sqlalchemy import func, select

    user = await _seed_user(db, "cascade@example.com")
    document = await _seed_document(db, user, provider, "Doomed", ["a", "b", "c"])
    await db.delete(document)
    await db.commit()

    remaining = await db.scalar(select(func.count(DocumentChunk.id)))
    assert remaining == 0
