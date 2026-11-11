# Semantic Search Engine

An incremental learning project for semantic search with PostgreSQL/pgvector,
sentence-transformers, and FastAPI. The current implementation provides a local
PostgreSQL/pgvector database and an **installable ingestion library with four
public functions**. Each document produces one vector and one database row:
there is no chunking, summarization, or content rewriting. API endpoints,
hybrid search, and ranking evaluation are not implemented yet.

## Current files

| File | Purpose |
| --- | --- |
| [`compose.yaml`](compose.yaml) | PostgreSQL with pgvector, persistent storage, localhost port, and readiness probe. |
| [`init.sql`](init.sql) | Enable vector support and create the documents table and metadata index. |
| [`pyproject.toml`](pyproject.toml) | Independent, installable Python project with a src-layout package and direct dependencies. |
| [`src/semantic_search_engine/`](src/semantic_search_engine/) | Four public ingestion functions and a thin standalone module entry point. |
| [`tests/`](tests/) | Offline unit tests and opt-in isolated PostgreSQL/model checks. |
| [`uv.lock`](uv.lock) | Resolved dependency versions for reproducible setup; keep in Git. |
| [`.python-version`](.python-version) | Preferred Python version: 3.12. |

## Python environment with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
From this project directory:

```bash
uv sync --locked
uv run --locked python --version
```

uv creates this project's `.venv/`; activation is optional when using `uv run`.
Do not reuse the root or lab environment. From the repository root, the equivalent is:

```bash
uv sync --locked --project src/semantic-search-enine
uv run --locked --project src/semantic-search-enine python --version
```

| Dependency | Role |
| --- | --- |
| `fastapi` | REST API framework. |
| `uvicorn[standard]` | ASGI server for the future API. |
| `sentence-transformers` | Generate embeddings locally, including 384-D MiniLM vectors. |
| `asyncpg` | Async PostgreSQL connections and queries. |
| `pgvector` | Python vector adapters; does not install the PostgreSQL extension. The ingestion SQL currently uses parameterized vector text, so no custom connection codec is required. |
| `numpy` | Validate vector shape, finite values, and float32 conversion. |

The lockfile belongs to this project; the repository root is not a uv workspace.
Commit `pyproject.toml`, `uv.lock`, and `.python-version`, not `.venv/` or credentials.
Use `uv add <package>` here for intentional dependency changes. PyTorch and its
transitive dependencies can require several GB; installing packages does not
load model weights, run an API, or start PostgreSQL.

## Start the database

Requires Docker Engine and Docker Compose. From this directory:

```bash
# Bash: enter a local database password without recording it in shell history.
read -r -s -p "PostgreSQL password: " POSTGRES_PASSWORD
printf '\n'
export POSTGRES_PASSWORD

# Validate without starting anything, then start when ready.
docker compose config --quiet
docker compose up -d --wait
```

Alternatively, set `POSTGRES_PASSWORD` in a local, Git-ignored `.env` beside
`compose.yaml`. Never commit real credentials. Avoid sharing the output of
`docker compose config` without `--quiet`, since it can include resolved secrets.
The password is required: the configuration has no committed default password.

The first start downloads `pgvector/pgvector:pg16` if necessary and creates
persistent data. Connection defaults are:

| Setting | Default |
| --- | --- |
| Host | `127.0.0.1` (not exposed to other machines) |
| Port | `5432`; override with `POSTGRES_PORT` if occupied |
| Database | `semantic_search`; override with `POSTGRES_DB` |
| User | `semantic_search`; override with `POSTGRES_USER` |
| Password | Your `POSTGRES_PASSWORD` |

Connect inside the container and inspect the schema:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Inside psql, use `\dx` to inspect extensions, `\d documents` for the table,
and `\q` to exit. The healthcheck uses `pg_isready` to test server readiness;
it does not verify application credentials or table contents.

## Why the two volumes?

```yaml
volumes:
  - pgdata:/var/lib/postgresql/data
  - ./init.sql:/docker-entrypoint-initdb.d/init.sql
```

- `pgdata` is a Docker-managed named volume holding PostgreSQL's actual database
  files. Data survives container replacement and ordinary `docker compose down`.
  Compose normally scopes its name with the project name.
- `init.sql` is a bind mount: Docker makes the repository's SQL file available
  at PostgreSQL's initialization directory. The image executes it **only when
  initializing an empty data directory**, not on every start.

Editing `init.sql` after initialization does not update an existing database.
Future changes need migrations or an explicitly authorized reset. Do not use
`docker compose down -v` casually: it deletes the named volume and its data.
Changing initialization credentials also does not change users/passwords already
stored in an existing volume.

## What the SQL does

| SQL | Meaning |
| --- | --- |
| `CREATE EXTENSION IF NOT EXISTS vector` | Enable pgvector's vector type, operators, and indexing capabilities in this database. |
| `id BIGSERIAL PRIMARY KEY` | Assign an auto-incrementing unique record identifier. |
| `title TEXT NOT NULL UNIQUE` | Require a title and prevent duplicate titles across the whole table. |
| `content TEXT NOT NULL` | Store the searchable text. |
| `embedding vector(384)` | Reserve a 384-dimensional vector, matching `all-MiniLM-L6-v2`; other dimensions are rejected. |
| `metadata JSONB DEFAULT '{}'` | Store structured attributes such as topic, source, or language; default to an empty object when omitted. |
| `created_at` / `updated_at` | Default timestamps to insertion time. `updated_at` does **not** update automatically; later update logic must maintain it. |
| GIN index on `metadata` | Accelerate compatible JSONB queries such as containment with `@>`; not all JSON expressions use this index. |

The schema is preserved as requested: embeddings and metadata can still be
explicitly NULL. There is **no vector ANN index yet**; the GIN index handles
metadata, not vector similarity. `CREATE TABLE` and `CREATE INDEX` are initial
DDL, not a rerunnable migration. This ingestion step uses the global unique title
as document identity and explicitly updates `updated_at` on changed records.

## Ingestion: four public functions

```python
from semantic_search_engine import (
    load_document,
    generate_embedding,
    upsert_document,
    ingest,
)
```

| Function | Responsibility |
| --- | --- |
| `load_document(path, *, root=None)` | Read one UTF-8 `.txt` or `.md` file into a dictionary with `title`, unchanged `content`, and `metadata`. With `root`, its relative path is its title and `metadata.source`; otherwise its filename is used. |
| `generate_embedding(texts, *, model=None, cache_folder=None)` | Encode complete texts into a normalized float32 array shaped `(n, 384)`, with `MODEL_NAME = "all-MiniLM-L6-v2"` and `BATCH_SIZE = 64`. Lazily loads the local model or reuses one supplied by the caller. |
| `await upsert_document(connection, document, embedding)` | Parameterized insert/update on `title`. Replaces content/vector/metadata, preserves `id` and `created_at`, and sets `updated_at`. Caller owns the connection and any transaction. |
| `await ingest(documents, connection, *, model=None, cache_folder=None)` | Validate a batch, read existing rows, embed only new/changed content, and atomically upsert changes. Returns `inserted`, `updated`, `unchanged`, and `embedded` counts. |

A library document is `{"title": "manuals/guide.md", "content": "...", "metadata": {}}`.
Metadata is optional and defaults to `{}`. Titles/content must be nonblank;
metadata must be a JSON object with valid finite JSON values. Duplicate titles
in one batch and invalid vector dimensions/values are rejected. These functions
do not create the database or table: initialize them with Compose first.

### Run independently

From this directory, after starting PostgreSQL and running `uv sync --locked`:

```bash
# Reuse the POSTGRES_PASSWORD exported when starting Compose.
# If using a new shell, enter that same database password privately:
read -r -s -p "PostgreSQL password: " POSTGRES_PASSWORD
printf '\n'
export POSTGRES_PASSWORD

uv run --locked python -m semantic_search_engine ./documents --model-cache .cache/models
```

Without `DATABASE_URL`, the command uses `127.0.0.1`, port `5432`, database
`semantic_search`, and user `semantic_search`, exactly matching Compose defaults.
It accepts the same `POSTGRES_USER`, `POSTGRES_DB`, and `POSTGRES_PORT` overrides,
and safely URL-encodes the password. `DATABASE_URL` takes precedence if explicitly
set; unset an old value to use the local Compose settings. There is no hardcoded
password. The command reads exported environment variables, not `.env` files.
If Compose reads a local `.env`, export those same values in the ingestion shell.

Create `./documents` and put your own UTF-8 `.txt`/`.md` files inside it; nested
folders are supported. Other extensions are ignored. Missing/empty folders,
blank documents, and files resolving outside the source root are rejected.
No model is loaded and no database is contacted on `--help`:

```bash
uv run --locked python -m semantic_search_engine --help
```

The command prints a JSON summary to stdout, for example:

```json
{"embedded": 2, "inserted": 2, "unchanged": 0, "updated": 0}
```

Exit codes: `0` success; `2` argument/input/absent connection settings errors;
`1` database/model/ingestion failure. Connection details and document contents are not included in CLI error
messages. The library propagates exceptions so the host system can handle them.
Importing it does not connect, download weights, or start a service.

### Integrate into another system

Install this project into the host system's environment, for example with
`uv add --editable /path/to/AI-engineering/src/semantic-search-enine`. Your
application can then call the functions without using this CLI or FastAPI:

```python
import asyncio
import os
from pathlib import Path

import asyncpg
from semantic_search_engine import ingest, load_document

async def main():
    root = Path("documents")
    documents = [load_document(root / "guide.md", root=root)]
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        stats = await ingest(documents, connection, cache_folder=".cache/models")
        print(stats)
    finally:
        await connection.close()

asyncio.run(main())
```

A host may also construct document dictionaries directly, or call
`generate_embedding([document["content"]])[0]` then `await upsert_document(...)`.
Pass a preloaded compatible model to reuse it across ingestion calls. The caller
retains connection/model ownership; embeddings are offloaded from the async event
loop when `ingest` calls the synchronous encoder.

### Incremental behavior and limits

- New title: generate an embedding and insert a row.
- Same title, changed content or missing embedding: generate and upsert its vector.
- Only metadata changed: update it using the stored vector; no model call.
- Everything unchanged: no write, timestamp change, or model call.
- All writes in one `ingest` call share a transaction. A failure rolls back that
  batch; PostgreSQL sequences may still have gaps after conflicts or rollbacks.
- Metadata is replaced, not merged. Deleted/renamed files do not remove old rows.
- Relative paths define identity within a shared namespace: two unrelated source
  roots with the same relative filename target the same database title. Hosts
  must supply distinct titles if they need separate namespaces.
- This is a single-writer, in-memory batch implementation, not a streaming or
  concurrent-ingestion coordinator. Counters reflect the initially read snapshot.
- Keep one compatible embedding model for the collection. Model identity/version
  is not stored; changing it requires a separately planned full re-embedding.
- Text is passed as-is, once per document, with no chunk strategy. The model's
  tokenizer still has its own maximum sequence length and truncates long inputs;
  passing a full document does not make its context window unlimited.
- No OpenAI calls or API key are needed. Local inference has no per-call API
  charge, but uses your CPU/GPU, RAM, disk, and electricity. First actual model
  use downloads weights unless cached; use `--model-cache` to control their location.
- `BATCH_SIZE = 64` controls how many complete documents the encoder processes
  together; it does not split a document into chunks.

## Tests

From this directory:

```bash
# Offline: mock models/database, real file loading and vector validation.
uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v
```

PostgreSQL tests are skipped unless `SEMANTIC_SEARCH_TEST_DSN` is explicitly
set to a **disposable database**. They create/drop their own random schemas.
Never point them at a production database. To run them against your test server:

```bash
export SEMANTIC_SEARCH_TEST_DSN="$DATABASE_URL"  # ONLY if this is disposable
uv run --locked python -m unittest discover -s tests -p 'test_postgres.py' -v
```

Set `SEMANTIC_SEARCH_REAL_MODEL=1` to opt in to the real MiniLM integration test
and optionally `SEMANTIC_SEARCH_MODEL_CACHE` to its cache directory. Otherwise
that check is skipped; the remaining PostgreSQL tests use deterministic vectors.

## Scope and next steps

This is a local development setup, **not a production deployment**: it has no
backup strategy, TLS, least-privilege application role, or operational policies.
The image's `POSTGRES_USER` is an initialization administrator, not a restricted
application account. Docker and pgvector do not require an API subscription,
but hosting, storage, and resource usage are not guaranteed free.

Ingestion is available as a library and module command. Search, hybrid retrieval,
evaluation, and API endpoints remain separate, explicitly requested steps.

To stop containers while retaining data:

```bash
docker compose down
```
