# From notebook to service

The original project was a single Jupyter notebook, `simplerag.ipynb`: load a
document, split it, embed it into FAISS, and run a retrieval chain against
Gemini. The retrieval idea survives unchanged. What changed is everything a
notebook can leave implicit.

## Cell by cell

| Notebook                                                    | Where it lives now                        | Why it changed |
| ----------------------------------------------------------- | ----------------------------------------- | -------------- |
| `TextLoader` / `PyPDFLoader` / `WebBaseLoader` + `bs4`       | `app/services/extraction.py`              | One entry point per source type, with size and MIME validation, and HTML stripped by the standard library instead of BeautifulSoup |
| `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)` | `app/services/chunking.py`         | Same recursive-separator algorithm, no dependency, and unit-tested. Defaults raised to 1000/150 — see below |
| `GoogleGenerativeAIEmbeddings(model="models/embedding-001")` | `app/providers/gemini.py`                 | Direct REST call with retries, timeouts, batching, and a dimension check |
| `FAISS.from_documents(...)` / `Chroma(persist_directory=…)`  | `document_chunks` + `app/services/retrieval.py` | Durable, shared between replicas, and filterable by owner in the same query |
| `db.as_retriever()`                                          | `retrieval.search(...)`                   | Ownership and `status = ready` are enforced in SQL, not after the fact |
| `ChatPromptTemplate.from_template("Answer the following…")`  | `app/services/rag.py`                     | Numbered, citable sources and an explicit instruction to decline when the context does not cover the question |
| `create_stuff_documents_chain` + `create_retrieval_chain`    | `app/api/v1/chat.py`                      | An HTTP endpoint that persists both turns and their citations in one transaction |
| `os.environ["GOOGLE_API_KEY"] = "…"` in a cell               | `app/core/config.py` + `.env`             | Typed settings, no secret in source control |
| Reading the answer out of a cell                             | React SPA with streaming and citations     | |

## What the notebook did not have to answer

**Whose documents are these?** A notebook has one user. A service needs
accounts, ownership on every row, and queries that cannot return another user's
data. Retrieval filters by `owner_id` inside the ANN query itself.

**What happens while the embedding runs?** In a notebook you wait for the cell.
Over HTTP you cannot hold a request open for a 40-page PDF, so ingestion became
a state machine on the document row with a background task and a status the
client polls.

**What if the provider fails?** A notebook shows a traceback. A service records
the reason on the row, returns `502` with a readable message, retries transient
upstream errors with backoff, and offers a reingest endpoint.

**What is true after a restart?** FAISS rebuilt from memory is not. Vectors
live in Postgres next to the documents they belong to, under the same
transaction.

**How do you know it still works?** A notebook is verified by looking at it.
The suite here runs on every push against a real pgvector database with a
deterministic provider, covering ingestion, ranking, ownership isolation, token
rotation, and streaming.

## Chunk size

The notebook used `chunk_size=200, chunk_overlap=50`. That works over one page
of text, but 200 characters is roughly a sentence and a half: sentences get
split mid-clause, each chunk costs an embedding call, and retrieved passages
are often too small to answer anything on their own.

The defaults here are 1000 / 150. Passages stay coherent enough to quote in a
citation, the number of embedding calls drops by about five times, and the
overlap still carries a fact that straddles a boundary into one whole chunk.
Both values are configurable (`CHUNK_SIZE`, `CHUNK_OVERLAP`); changing them
means re-ingesting, which is what `POST /documents/{id}/reingest` is for.

## Embedding model

The notebook used `models/embedding-001`. The default here is
`models/text-embedding-004`, also 768-dimensional, so the schema is unchanged.
Set `GEMINI_EMBEDDING_MODEL` back if you want the original. Any model with a
different output width needs `EMBEDDING_DIM` changed and a migration —
`docs/architecture.md` has the recipe.

## What the notebook still does better

It is eleven cells long and you can read all of it in a minute. If the goal is
to try an idea, a notebook is the right tool. The rewrite is worth it only once
the thing has to keep running when you are not watching.
