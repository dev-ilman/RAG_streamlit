# Architecture

## Data model

Six tables. Everything a user owns hangs off `users` with `ON DELETE CASCADE`,
so deleting an account removes their corpus and history in one statement.

```
users ──┬── refresh_tokens        one row per issued session, revocable
        ├── documents ── document_chunks (embedding vector(768))
        ├── document_chunks       denormalised owner_id, see below
        └── chat_sessions ── chat_messages (citations jsonb)
```

### `users`

`email` is unique and indexed. Passwords are bcrypt hashes; nothing else about
the credential is stored.

### `refresh_tokens`

A refresh token is an opaque random string, and only its SHA-256 hash is
persisted. That has three consequences worth having:

1. A database leak yields no usable tokens.
2. Sign-out is real — the row is marked revoked, not just forgotten by the
   browser.
3. Rotation is enforceable: `/auth/refresh` revokes the row it consumed, so a
   replayed token is rejected.

A stateless refresh JWT cannot do any of these.

### `documents`

Carries the ingestion state machine (`pending → processing → ready|failed`) and
the failure reason, so the UI can show why something did not index. The unique
constraint on `(owner_id, checksum)` makes re-uploading the same bytes a 409
instead of silently doubling the corpus.

### `document_chunks`

One row per embedded passage.

`owner_id` is duplicated from `documents` deliberately. Every retrieval query
filters by owner, and holding the column here lets Postgres apply that filter
without joining back to `documents` on the hot path. The cost is one extra UUID
per chunk; the benefit is a simpler, faster query and one less way to
accidentally leak another user's data.

The vector column is indexed with HNSW under `vector_cosine_ops`:

```sql
CREATE INDEX ix_document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

HNSW rather than IVFFlat because IVFFlat needs a training pass over existing
rows to build its lists, and the corpus starts empty.

### `chat_messages`

`citations` is a JSONB array snapshotted at answer time — chunk id, document
id, title, similarity score, and the excerpt. Storing it rather than
re-deriving it means an old answer can still be explained after the underlying
document has been edited or deleted.

## Request flow

### Ingestion

```
POST /documents/upload
  ├─ validate size and type            → 413 / 415
  ├─ sha256 the bytes, check for dupes → 409
  ├─ write the file, insert the row as `pending`
  ├─ return 202 with the document
  └─ background task:
       processing → extract text → split into chunks
                  → embed in batches → insert chunks
                  → ready   (or failed, with the reason on the row)
```

The client polls the document until it reaches a terminal state. Running the
pipeline in-process keeps the deployment to two containers; the seam for
scaling out is `ingest_document`, which already takes its own session and can
be driven by a Celery or ARQ worker without touching anything else.

### Question answering

```
POST /chat/sessions/{id}/ask
  ├─ load the last N turns for conversational context
  ├─ embed the question (RETRIEVAL_QUERY task type)
  ├─ ANN search: cosine distance, filtered by owner and status=ready
  ├─ assemble a numbered context block into the system prompt
  ├─ call the model
  └─ persist both turns plus the citations in one transaction
```

With no relevant passages the model is never called: the endpoint returns a
fixed "not in your documents" answer. That saves a token spend and removes the
most common path to a hallucinated reply.

## Design decisions

### pgvector instead of FAISS

FAISS is an in-process index. It is rebuilt on every start, lives in one
process's memory, cannot be filtered by user without a second data structure,
and has no transactional relationship with the rest of the data.

Putting the vectors in Postgres makes retrieval durable, shared across API
replicas, consistent with the documents they belong to, and — most usefully —
filterable: ownership and document scoping are ordinary SQL predicates applied
by the same query that does the similarity ranking.

### No LangChain

The notebook used LangChain for loaders, splitting, the vector store, and the
chain. Each of those is a handful of lines here:

| LangChain                        | Replacement                          |
| -------------------------------- | ------------------------------------ |
| `PyPDFLoader` / `TextLoader` / `WebBaseLoader` | `app/services/extraction.py` (pypdf + stdlib HTML parser) |
| `RecursiveCharacterTextSplitter` | `app/services/chunking.py`           |
| `FAISS` / `Chroma`               | pgvector + `app/services/retrieval.py` |
| `create_retrieval_chain`         | `app/services/rag.py`                |
| `GoogleGenerativeAI*`            | `app/providers/gemini.py` (REST)     |

The trade is a few hundred lines of our own code against a smaller dependency
tree, request shapes we can read, retry and timeout behaviour we control, and
a provider seam thin enough to fake in tests.

### Two tokens, two lifetimes

The access token is a 15-minute JWT held in JavaScript memory only. The refresh
token is an opaque 14-day value in an httpOnly cookie scoped to `/api/v1/auth`.

An XSS bug can therefore steal at most a token that expires in minutes, and
cannot read the one that would let it mint more. Nothing is kept in
`localStorage`. The SPA and the API are served from the same origin — Vite
proxies `/api` in development, nginx proxies it in production — so the cookie
works without any CORS credentials arrangement.

### Deterministic fake provider

`app/providers/fake.py` produces embeddings by hashing tokens into a fixed
vector space. Lexical overlap therefore yields real cosine similarity, so
retrieval assertions in CI test actual ranking behaviour rather than a stub
that always returns the first row. This is what allows the full ingestion and
retrieval pipeline to be exercised on every push with no API key.

### Changing the embedding dimension

`EMBEDDING_DIM` must match the width of the `vector` column. pgvector requires
a fixed dimension for an indexed column, so switching to a model with a
different output size means a migration:

```python
op.drop_index("ix_document_chunks_embedding_hnsw", table_name="document_chunks")
op.execute("DELETE FROM document_chunks")            # vectors are not portable
op.alter_column("document_chunks", "embedding", type_=Vector(NEW_DIM))
op.execute("CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)")
```

Existing documents then need re-ingesting (`POST /documents/{id}/reingest`).
The Gemini provider checks the returned width on every call and fails loudly
rather than writing vectors the column cannot hold.

## Operational surface

- `GET /health/live` touches nothing — an orchestrator should not restart a
  healthy container because the database blinked.
- `GET /health/ready` reports database connectivity and returns 503 when it is
  unreachable.
- Every response carries `x-request-id`, echoed from the request when present,
  and every log line is JSON carrying the same id.
- A fixed-window rate limiter keyed by client IP runs in-process. It is honest
  about its limits: behind more than one replica, move it to Redis or the
  platform gateway.
