# RAG Platform

A production-grade retrieval-augmented generation application: upload your own
documents, ask questions about them, and get answers that cite the exact
passages they came from.

This is a rebuild of a LangChain notebook as real software. The retrieval
pipeline is the same idea; everything around it — authentication, persistence,
API design, tests, containers, CI, deployment — is what turns it into a
product.

```
┌──────────────┐   /api    ┌──────────────┐   SQL + pgvector   ┌──────────────┐
│ React + TS   │ ────────► │   FastAPI    │ ─────────────────► │  PostgreSQL  │
│ (nginx/Vite) │ ◄──────── │   backend    │ ◄───────────────── │  + pgvector  │
└──────────────┘   SSE     └──────┬───────┘                    └──────────────┘
                                  │ HTTPS
                                  ▼
                         Gemini (embeddings + chat)
```

## What it does

- **Accounts** — register and sign in. Every document, chat, and message is
  scoped to its owner and invisible to everyone else.
- **Ingestion** — upload a PDF, text, Markdown, HTML, or CSV file; fetch a web
  page by URL; or paste text. The file is extracted, chunked, embedded, and
  indexed in the background, with live status in the UI.
- **Grounded answers** — questions are answered only from retrieved passages,
  and every reply carries its sources with similarity scores.
- **Streaming** — answers arrive token by token over server-sent events.
- **History** — chat sessions and messages persist, including the citations
  that produced each answer.

## Authentication

Short-lived, stateless access tokens plus rotating, single-use refresh tokens:

| Token   | Form                          | Lifetime | Where it lives                    |
| ------- | ----------------------------- | -------- | --------------------------------- |
| Access  | JWT (HS256), `Authorization: Bearer` | 15 min | Browser memory only        |
| Refresh | Opaque random string          | 14 days  | httpOnly, SameSite=lax cookie; only its SHA-256 hash is stored in Postgres |

- Passwords are hashed with bcrypt; login does a dummy hash check for unknown
  emails so response time does not reveal which addresses are registered.
- On a 401 the frontend silently calls `/auth/refresh` once and replays the
  request; each refresh revokes the old token and issues a new pair.
- Every query is scoped to the user taken from the verified token, so users only
  ever see their own documents and chats.

## Stack

| Layer     | Choice                                                            |
| --------- | ----------------------------------------------------------------- |
| Backend   | FastAPI, SQLAlchemy 2 (async), Alembic, Pydantic v2                |
| Database  | PostgreSQL 16 + pgvector, HNSW cosine index                        |
| Auth      | JWT access tokens + rotating refresh tokens in an httpOnly cookie  |
| Frontend  | React 18, TypeScript (strict), Vite, React Router                  |
| Models    | Gemini over REST, behind a provider interface with an offline fake |
| Infra     | Docker multi-stage builds, GitHub Actions, Azure or AWS free tier  |

## Quick start

```bash
git clone https://github.com/dev-ilman/RAG_streamlit.git rag-platform && cd rag-platform
cp .env.example .env
```

Put a [Google AI Studio](https://aistudio.google.com/apikey) key in
`GOOGLE_API_KEY`, then:

```bash
docker compose up --build
```

- App — <http://localhost:5173>
- API docs — <http://localhost:8000/docs>
- Health — <http://localhost:8000/api/v1/health/ready>

To load a demo account (`demo@example.com` / `demo-password-123`) and one
sample document:

```bash
docker compose run --rm backend python -m app.cli seed
```

### No API key?

The whole application runs offline with a deterministic stand-in provider —
the same one CI uses. Retrieval is genuinely exercised; only the wording of the
final answer is synthetic.

```bash
LLM_PROVIDER=fake docker compose up --build
```

## Running tests

```bash
make test
```

Or directly:

```bash
cd backend  && pytest                    # needs a pgvector database
cd frontend && npm run test -- --run
```

The backend suite runs against a real PostgreSQL with pgvector rather than
SQLite, because the behaviour most worth testing — similarity ranking, unique
constraints, cascade deletes — does not exist in SQLite. The model provider is
swapped for a deterministic fake, so the suite needs no API key and no network.

```bash
docker run -d --name ragtest-db -p 5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  pgvector/pgvector:pg16
```

## Repository layout

```
backend/
  app/
    api/v1/       auth, documents, chat, health endpoints
    core/         config, security, deps, middleware, errors, logging
    db/           declarative base, async engine and session
    models/       User, RefreshToken, Document, DocumentChunk, ChatSession, ChatMessage
    providers/    model interface + Gemini (REST) and deterministic fake
    schemas/      request and response models
    services/     chunking, extraction, ingestion, retrieval, prompt assembly
  alembic/        migrations, including the pgvector extension and HNSW index
  tests/          pytest suite (unit + integration against pgvector)
frontend/
  src/api/        typed fetch client, token refresh, SSE stream reader
  src/components/ layout, route guard, message and citation rendering
  src/context/    auth state
  src/pages/      login, register, documents, chat
  src/test/       vitest suite
infra/            Azure Container Apps and AWS EC2 deployment
scripts/          end-to-end smoke test
docs/             architecture, API reference, deployment, notebook migration
```

## Continuous integration

Every push runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job        | What it does                                                          |
| ---------- | --------------------------------------------------------------------- |
| `backend`  | ruff lint + format, mypy, migrations apply, model/migration drift, pytest with coverage |
| `frontend` | eslint, `tsc -b`, vitest, production build                             |
| `docker`   | builds both production images with layer caching                       |
| `smoke`    | boots the full compose stack and drives register → ingest → ask over HTTP |

`.github/workflows/deploy.yml` publishes images to GHCR on `main` and rolls out
Azure Container Apps revisions when the Azure variables are configured.

## Deployment

Two paths, both inside a free tier — see [docs/deployment.md](docs/deployment.md):

- **Azure** (recommended): Container Apps with scale-to-zero plus PostgreSQL
  Flexible Server B1ms. One script: `infra/azure/provision.sh`.
- **AWS**: a single t2.micro/t3.micro EC2 instance running the production
  compose stack, via `infra/aws/user-data.sh`.

## Documentation

- [Architecture](docs/architecture.md) — data model, request flow, and the
  reasoning behind the design decisions.
- [API reference](docs/api.md) — every endpoint with examples.
- [Deployment](docs/deployment.md) — Azure and AWS, step by step.
- [From notebook to service](docs/migration-from-notebook.md) — what each cell
  of the original notebook became, and why.

## Configuration

All settings come from the environment; see [.env.example](.env.example). The
ones worth knowing:

| Variable                   | Default                      | Notes                                         |
| -------------------------- | ---------------------------- | --------------------------------------------- |
| `SECRET_KEY`               | dev placeholder              | Required in production; startup refuses the placeholder |
| `LLM_PROVIDER`             | `gemini`                     | `fake` runs the stack with no API key         |
| `GEMINI_EMBEDDING_MODEL`   | `models/gemini-embedding-001` | Must produce `EMBEDDING_DIM` values          |
| `EMBEDDING_DIM`            | `768`                        | Must match the vector column; changing it needs a migration |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150`           | Re-ingest documents after changing            |
| `RETRIEVAL_TOP_K`          | `5`                          | Passages sent to the model per question       |
| `COOKIE_SECURE`            | `false`                      | Set `true` when serving over HTTPS            |

## Licence

MIT — see [LICENSE](LICENSE).
