"""Small operational CLI: ``python -m app.cli <command>``."""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.core.config import settings
from app.core.security import hash_password
from app.db.session import SessionLocal, dispose_engine
from app.models import SourceType, User
from app.providers import build_provider
from app.services import ingestion

DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-password-123"

SAMPLE_TITLE = "Retrieval-Augmented Generation: a short primer"
SAMPLE_TEXT = """\
Retrieval-augmented generation (RAG) combines a document retriever with a
generative language model. Rather than relying on facts memorised during
pre-training, the system searches a corpus at query time and conditions the
model on the passages it finds.

A RAG pipeline has four stages. Ingestion loads source documents and extracts
plain text. Chunking splits that text into passages small enough to embed and
large enough to stay coherent. Embedding maps each passage to a dense vector.
Retrieval embeds the user question with the same model and returns the nearest
passages by cosine similarity.

Chunk size is the main tuning knob. Small chunks give precise matches but
fragment arguments across boundaries; large chunks preserve context but dilute
the embedding, so a single relevant sentence can be outweighed by surrounding
text. Overlapping chunks mitigate boundary effects at the cost of storage.

Storing vectors in PostgreSQL with the pgvector extension keeps embeddings in
the same transactional store as the application's other data. An HNSW index
provides approximate nearest-neighbour search, and ordinary SQL predicates can
filter by owner or document before the similarity ranking is applied.

Grounding is what separates RAG from a plain chatbot. Because every answer is
produced from retrieved passages, each claim can be traced back to a source,
and the model can be instructed to decline when the corpus does not cover the
question.
"""


async def seed() -> None:
    """Create the demo user and ingest one sample document. Idempotent."""
    provider = build_provider()
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == DEMO_EMAIL))
        if user is None:
            user = User(
                email=DEMO_EMAIL,
                hashed_password=hash_password(DEMO_PASSWORD),
                full_name="Demo User",
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            print(f"created user {DEMO_EMAIL} / {DEMO_PASSWORD}")
        else:
            print(f"user {DEMO_EMAIL} already exists")

        try:
            document = await ingestion.create_document(
                db,
                owner_id=user.id,
                title=SAMPLE_TITLE,
                data=SAMPLE_TEXT.encode("utf-8"),
                mime_type="text/plain",
                source_type=SourceType.TEXT,
            )
        except ingestion.DuplicateDocumentError:
            print("sample document already ingested")
        else:
            await ingestion.ingest_document(document.id, provider, raw_text=SAMPLE_TEXT)
            print(f"ingested sample document {document.id}")

    await provider.aclose()
    await dispose_engine()


async def create_user(email: str, password: str) -> None:
    async with SessionLocal() as db:
        if await db.scalar(select(User).where(User.email == email.lower())):
            print(f"user {email} already exists", file=sys.stderr)
            return
        db.add(User(email=email.lower(), hashed_password=hash_password(password)))
        await db.commit()
        print(f"created {email}")
    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli", description="RAG platform admin CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="create the demo user and sample document")
    sub.add_parser("config", help="print the effective configuration")
    create = sub.add_parser("create-user", help="create a user")
    create.add_argument("email")
    create.add_argument("password")

    args = parser.parse_args()
    if args.command == "seed":
        asyncio.run(seed())
    elif args.command == "create-user":
        asyncio.run(create_user(args.email, args.password))
    else:
        for key, value in settings.model_dump().items():
            if any(s in key for s in ("password", "secret", "api_key")):
                value = "***"
            print(f"{key} = {value}")


if __name__ == "__main__":
    main()
