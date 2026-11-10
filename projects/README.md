# Projects

Standalone learning projects, separate from the focused examples and labs.
Each project documents its own setup, implemented features, and limitations.
Python projects use independent uv-managed environments: commit `pyproject.toml`
and `uv.lock`, never `.venv`, `.env`, or local caches. Infrastructure-only first
steps may precede their explicitly requested Python setup.

| Project | Purpose | Current scope |
| --- | --- | --- |
| [Semantic Search Engine](semantic-search-engine/) | Learn end-to-end semantic retrieval with pgvector, sentence-transformers, and FastAPI. | Local PostgreSQL 16 + pgvector Compose configuration and initial document schema. Python ingestion, API, hybrid search, and evaluation are not implemented yet. |
