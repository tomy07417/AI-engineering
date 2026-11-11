# Source Index

Source modules and learning projects live here. Each independent Python project
has its own uv-managed environment: commit `pyproject.toml` and `uv.lock`, never
`.venv`, `.env`, or local caches. These projects are not members of a shared uv
workspace.

| Directory | Purpose | Current scope |
| --- | --- | --- |
| [Semantic Search Engine](semantic-search-enine/) | Learn end-to-end semantic retrieval with pgvector, sentence-transformers, and FastAPI. | PostgreSQL 16 + pgvector and an installable library with four ingestion functions: UTF-8 text loading, MiniLM embeddings, async upsert, and incremental ingestion. CLI available; no chunking, API, search, or evaluation yet. |
| [Labs](labs/) | Small guided experiments. | The embedding explorer compares models, category similarities, and t-SNE/UMAP projections; see the root README for its workflow. |
| [AI service scaffold](ai_service/) | Existing shared Pydantic models. | Not a runnable service. |

For the semantic search project, run `uv sync --locked` inside its directory,
or from the repository root:

```bash
uv sync --locked --project src/semantic-search-enine
uv run --locked --project src/semantic-search-enine python --version
```

The directory spelling `semantic-search-enine` follows the requested location;
the Python project distribution name remains `semantic-search-engine`.
