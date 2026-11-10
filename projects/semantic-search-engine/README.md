# Semantic Search Engine

An incremental learning project for semantic search with PostgreSQL/pgvector,
sentence-transformers, and FastAPI. The current step provides **only a local
PostgreSQL 16 database and its initial schema**. Ingestion, API endpoints,
hybrid search, and ranking evaluation are not implemented yet.

## Current files

| File | Purpose |
| --- | --- |
| [`compose.yaml`](compose.yaml) | PostgreSQL with pgvector, persistent storage, localhost port, and readiness probe. |
| [`init.sql`](init.sql) | Enable vector support and create the documents table and metadata index. |

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
DDL, not a rerunnable migration. The global unique title constraint also matters
when a later ingestion step starts storing multiple chunks from one document.

## Scope and next steps

This is a local development setup, **not a production deployment**: it has no
backup strategy, TLS, least-privilege application role, or operational policies.
The image's `POSTGRES_USER` is an initialization administrator, not a restricted
application account. Docker and pgvector do not require an API subscription,
but hosting, storage, and resource usage are not guaranteed free.

Python setup will be added when explicitly requested: this project will use
its own `pyproject.toml`, committed `uv.lock`, and Git-ignored `.venv` managed
with uv. This database-only step does not create Python code or an environment.

To stop containers while retaining data:

```bash
docker compose down
```
