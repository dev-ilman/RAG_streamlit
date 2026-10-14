# API reference

Base path `/api/v1`. Interactive docs are served at `/docs`, and the generated
schema at `/openapi.json`.

## Conventions

Authenticated requests carry the access token:

```
Authorization: Bearer <access_token>
```

Every error uses one envelope:

```json
{ "error": { "code": "http_404", "message": "Document not found" } }
```

| Status | When                                                        |
| ------ | ----------------------------------------------------------- |
| 401    | Missing, invalid, or expired access token                   |
| 403    | Authenticated but the account is disabled                   |
| 404    | Not found, or owned by another user (existence is not confirmed) |
| 409    | Duplicate email, duplicate document, or a conflicting state |
| 413    | Upload exceeds `MAX_UPLOAD_BYTES`                           |
| 415    | Unsupported file type                                       |
| 422    | Request body failed validation                              |
| 429    | Rate limit exceeded; see `Retry-After`                      |
| 502    | The model provider failed                                   |

## Auth

### `POST /auth/register`

```json
{ "email": "you@example.com", "password": "at-least-ten-chars", "full_name": "You" }
```

`201` with an access token, and a rotating refresh token set as an httpOnly
cookie.

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": { "id": "…", "email": "you@example.com", "full_name": "You", "is_active": true, "created_at": "…" }
}
```

### `POST /auth/login`

Same body minus `full_name`, same response. Unknown addresses and wrong
passwords both return `401` after the same amount of work, so the endpoint does
not reveal which accounts exist.

### `POST /auth/refresh`

Takes no body; reads the refresh cookie. Returns a fresh access token and
replaces the cookie. The consumed refresh token is revoked, so replaying it
returns `401`.

### `POST /auth/logout`

Revokes the refresh token server-side and clears the cookie. `204`.

### `GET /auth/me`

The authenticated user.

## Documents

### `GET /documents?limit=50&offset=0&document_status=ready`

```json
{ "items": [ … ], "total": 3, "limit": 50, "offset": 0 }
```

### `POST /documents/upload`

`multipart/form-data` with `file`, and an optional `title`. Accepts PDF, TXT,
Markdown, HTML, CSV, and JSON. Returns `202` with the document in `pending`;
poll it until `status` is `ready` or `failed`.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@paper.pdf" -F "title=Ageing mechanisms"
```

### `POST /documents/url`

```json
{ "url": "https://example.com/article", "title": "Optional" }
```

Fetches the page, strips scripts, styles, and markup, and indexes the text.

### `POST /documents/text`

```json
{ "title": "Meeting notes", "content": "…" }
```

### `GET /documents/{id}`

```json
{
  "id": "…", "title": "Ageing mechanisms", "status": "ready",
  "source_type": "upload", "chunk_count": 42, "size_bytes": 184320,
  "error": null, "created_at": "…", "updated_at": "…"
}
```

### `POST /documents/{id}/reingest`

Re-runs the pipeline; useful after a transient provider failure or a change to
`CHUNK_SIZE`. `409` while the document is already processing.

### `DELETE /documents/{id}`

Deletes the document, its chunks, and the stored file. `204`.

### `POST /documents/search`

Retrieval without generation — the quickest way to see what the model would be
shown.

```json
{ "query": "what consumes cyclable lithium", "top_k": 5, "document_ids": null }
```

```json
{
  "query": "what consumes cyclable lithium",
  "hits": [
    {
      "chunk_id": "…", "document_id": "…", "document_title": "Ageing mechanisms",
      "chunk_index": 7, "score": 0.8421, "content": "Solid electrolyte interphase growth…"
    }
  ]
}
```

`score` is cosine similarity in `[0, 1]`; higher is closer.

## Chat

### `GET /chat/sessions`

Sessions, most recently active first.

### `POST /chat/sessions`

```json
{ "title": "Optional" }
```

`201`. The title is replaced by the first question asked, unless one was given.

### `GET /chat/sessions/{id}`

The session plus its full message history, oldest first.

### `PATCH /chat/sessions/{id}`

```json
{ "title": "Battery questions" }
```

### `DELETE /chat/sessions/{id}`

`204`. Messages cascade.

### `POST /chat/sessions/{id}/ask`

```json
{ "message": "What consumes cyclable lithium?", "top_k": 5, "document_ids": null }
```

Returns both persisted turns:

```json
{
  "session_id": "…",
  "user_message": { "role": "user", "content": "What consumes cyclable lithium?", … },
  "assistant_message": {
    "role": "assistant",
    "content": "Solid electrolyte interphase growth consumes cyclable lithium [1].",
    "citations": [
      { "chunk_id": "…", "document_id": "…", "document_title": "Ageing mechanisms",
        "chunk_index": 7, "score": 0.8421, "excerpt": "…" }
    ],
    "model": "models/gemini-2.5-flash",
    "latency_ms": 730
  }
}
```

When nothing relevant is found, `citations` is empty and the answer says so.

### `POST /chat/sessions/{id}/ask/stream`

Same body; responds with `text/event-stream`.

```
event: meta
data: {"session_id":"…","citations":[…]}

event: token
data: {"text":"Solid electrolyte "}

event: done
data: {"message_id":"…","latency_ms":730}
```

An `error` event carries `{"message": "…"}` if the provider fails mid-answer.
Both turns are persisted when the stream completes.

`EventSource` cannot be used to consume this: it only issues GET requests and
cannot set an `Authorization` header. The reference client reads the body with
`fetch` and a `ReadableStream` — see `frontend/src/api/chat.ts`.

## Health

| Endpoint        | Purpose                                                   |
| --------------- | --------------------------------------------------------- |
| `/health/live`  | Process liveness; touches no dependency                    |
| `/health/ready` | Reports database connectivity; `503` when it is unreachable |
