# AI Engineering: Examples and Labs

A hands-on collection for learning AI engineering with Python: provider SDKs,
async calls, streaming APIs, structured outputs, embeddings, and retrieval
experiments. Start with a small example, then use the embedding explorer to
inspect semantic similarity and two-dimensional projections.

**This is educational code, not a deployable production service.** Files named
`production` demonstrate patterns; they are not hardened implementations.
Provider calls can cost money and send your input to external services.

## Repository map

| Path | What you will find |
| --- | --- |
| [`examples/`](examples/) | Focused, mostly standalone demonstrations; see the complete catalog below. |
| [`src/`](src/README.md) | Source modules and independent projects; see the source index for setup and scope. |
| [`src/labs/embedding_explorer/`](src/labs/embedding_explorer/) | A guided comparison of OpenAI and local embeddings, with offline tests. |
| [`src/ai_service/models.py`](src/ai_service/models.py) | Pydantic v2 request, response, token usage, and discriminated-union models. |
| [`src/ai_service/__init__.py`](src/ai_service/__init__.py) | Empty package scaffold, not an application entry point. |
| [`main.py`](main.py) | A greeting-only entry point; it does not start an AI service. |
| [`pyproject.toml`](pyproject.toml) | Runtime dependencies, optional development tools, and tool settings. |
| [`scripts/`](scripts/) and [`tests/`](tests/) | Placeholder directories; actual tests currently live beside the examples and lab. |

The declared `serve` and `ingest` commands reference `ai_service.main:run` and
`ai_service.cli:ingest`, but those modules do not exist yet. Use the explicit
example and lab commands here instead. Packaging/service entry points remain
scaffolding despite the project metadata's “Production AI microservice” label.

## Projects

### [Semantic Search Engine](src/semantic-search-enine/)

Learn semantic retrieval with pgvector, sentence-transformers, and FastAPI.
Provides PostgreSQL + pgvector and an independent uv-managed ingestion library:
load complete `.txt`/`.md` documents, generate MiniLM embeddings, and incrementally
upsert records. The same four-function library runs through a thin CLI, without
chunking. API, search, and evaluation remain future steps.
See the [source index](src/README.md) for scope and setup links.

## Quick start: no API key required

Use **Python 3.12 or newer**; [`.python-version`](.python-version) selects 3.12.
Run commands from the repository root unless stated otherwise.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python examples/async_patterns.py
python main.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
`async_patterns.py` uses only the standard library and simulates provider calls;
`main.py` prints `Hello from ai-engineering!`.

### Install only what you need

The following installs the runtime dependencies declared in `pyproject.toml`
without relying on the unfinished package entry points:

```bash
python -m pip install 'fastapi>=0.115.0' 'uvicorn[standard]>=0.32.0' \
  'openai>=1.55.0' 'anthropic>=0.39.0' 'pydantic>=2.10.0' \
  'pydantic-settings>=2.6.0' 'httpx>=0.28.0' 'structlog>=24.4.0' \
  'chromadb>=0.5.0' 'tiktoken>=0.8.0'
```

Additional dependencies are not currently declared as a lab extra:

```bash
# Embedding explorer and numerical embedding examples
python -m pip install python-dotenv numpy sentence-transformers scikit-learn umap-learn matplotlib

# Only for the examples that need them
python -m pip install google-genai   # gemini_chat.py
python -m pip install tenacity       # httpx_client.py
python -m pip install Pillow         # multimodal_search.py image support

# Tests for the FastAPI example
python -m pip install pytest pytest-asyncio
```

The `dev` dependency group in `pyproject.toml` also lists pytest-cov, Ruff, mypy,
and respx for broader development tooling. ChromaDB is declared but is not
needed for the embedding explorer. Local sentence-transformers models can pull
in large dependencies, including PyTorch, and download weights on first use.
Plan for network access, disk space, and a slower first run; subsequent local
inference does not need a provider API key.

### Credentials and cost safety

- OpenAI examples use `OPENAI_API_KEY`; Anthropic uses `ANTHROPIC_API_KEY`;
  Gemini uses `GOOGLE_API_KEY`. Configure only providers you intend to call.
- A ChatGPT subscription **does not include OpenAI API billing**. Provider API
  access, quotas, supported models, and billing are separate concerns.
- Model IDs and SDK features in these examples may change or be unavailable to
  your account. Check access before running paid examples.
- Prefer environment variables supplied securely by your shell or secret manager.
  Most standalone SDK examples do **not** automatically load `.env` files.
- Never commit credentials, paste them into source, or include them in logs.
  A `.env` file is plaintext; `SecretStr` masks representations, not storage.
- Requests, retries, batches, and provider fallback can multiply costs. Use
  synthetic input, review the source, and configure provider-side usage limits.

## Examples: complete catalog

Unless a row says otherwise, run a demonstration with
`python examples/<filename>.py` from the root after installing its dependencies.
“OpenAI” and “Anthropic” below mean real requests with the corresponding key,
not simulated calls. Read a file's opening docstring before executing it.

### Async, HTTP, and configuration

| File | Concept | Execution notes |
| --- | --- | --- |
| [`async_patterns.py`](examples/async_patterns.py) | `TaskGroup` fan-out, semaphore-bounded batches, simulated word streaming. | Standard library only; no API calls. Concurrency bounds are not requests-per-minute limits. |
| [`httpx_client.py`](examples/httpx_client.py) | Connection pooling, timeouts, Tenacity retries, single-line SSE parsing. | Needs `tenacity`; main contacts OpenAI with a placeholder key and will not authenticate as-is. Supply credentials securely if adapting it; retrying POSTs may duplicate work/cost. |
| [`settings.py`](examples/settings.py) | Lazy cached Pydantic settings and masked secrets. | Requires an OpenAI key even when only displaying settings; reads root `.env` when run from the root. |
| [`errors_and_logging.py`](examples/errors_and_logging.py) | Exception hierarchy, request IDs, async logging decorator. | Direct execution only configures logging. Import and call `generate()` to exercise its simulated timeout; sanitize raw errors before real use. |
| [`token_budget.py`](examples/token_budget.py) | Token estimates, illustrative pricing, in-memory usage totals. | No inference calls; tiktoken may download encoding data. Prices are not verified current rates; budget checks happen after recording usage, not before spending. |

### Provider calls and prompting

| File | Concept | Execution notes |
| --- | --- | --- |
| [`openai_chat.py`](examples/openai_chat.py) | Async Chat Completions, usage, conversation history. | OpenAI; main runs a multi-turn conversation. |
| [`anthropic_chat.py`](examples/anthropic_chat.py) | Separate system prompts and extended thinking. | Anthropic; main calls both chat and thinking. Thinking needs supported SDK/model access and has a larger token budget. |
| [`gemini_chat.py`](examples/gemini_chat.py) | Gemini text generation, embeddings, PNG analysis helper. | Needs `google-genai` and Google key; main calls text and embedding APIs, not image analysis. The image helper uploads supplied PNG bytes. |
| [`multi_provider.py`](examples/multi_provider.py) | Provider interface and explicit model routing. | Initializes both OpenAI and Anthropic clients: both keys required. Basic text only, no Gemini implementation; Anthropic embedding method is unsupported. |
| [`prompt_patterns.py`](examples/prompt_patterns.py) | Step-by-step prompting, few-shot classification, answer voting. | OpenAI; voting is exact-string plurality, not a correctness guarantee. Concurrent samples have no rate/concurrency bound. |
| [`structured_outputs.py`](examples/structured_outputs.py) | OpenAI structured parsing with Pydantic enums and validation. | OpenAI only; refusals, API errors, and validation failures remain possible. Batch helper has no concurrency limit. |
| [`tool_calling.py`](examples/tool_calling.py) | Tool schemas, result messages, final response. | Real OpenAI calls but hardcoded tool results; one tool-call round, sequential execution. Validate arguments and authorization before connecting real systems. |

### Streaming, resilience, and web APIs

| File | Concept | Execution notes |
| --- | --- | --- |
| [`provider_streaming.py`](examples/provider_streaming.py) | OpenAI/Anthropic text accumulation, first-fragment latency, usage. | Both keys required; received fragments are not tokenizer tokens. |
| [`resilient_llm_client.py`](examples/resilient_llm_client.py) | Backoff, sequential sliding-window throttling, fallback. | Both keys required; fallback shares input with another provider. All exceptions are retried, SDK retries can multiply attempts, and limiter is not concurrency-safe. |
| [`fastapi_app.py`](examples/fastapi_app.py) | Lifespan-managed OpenAI client, dependency injection, `/chat`, `/health`. | Launch with Uvicorn below, not as a standalone script; OpenAI key required at startup. |
| [`sse_streaming.py`](examples/sse_streaming.py) | `/chat/stream` proxy with token, finish, usage, and error SSE events. | Launch with Uvicorn below; needs an exported OpenAI key. Raw exception messages are exposed: do not deploy publicly as-is. |
| [`test_fastapi_app.py`](examples/test_fastapi_app.py) | Async endpoint tests with mocked clients and settings. | Run explicitly with pytest (see Tests); no credentials or provider requests. Lifespan is not started by these tests. |

Start **one** local web example at a time:

```bash
python -m uvicorn examples.fastapi_app:app --host 127.0.0.1 --port 8000
# Or, for SSE:
python -m uvicorn examples.sse_streaming:app --host 127.0.0.1 --port 8000
```

For the first app, open `http://127.0.0.1:8000/docs` or `/health`. A `/chat`
request performs a paid provider call. The SSE app accepts POST JSON with
`message` at `/chat/stream` and terminates its event stream with `[DONE]`.
Neither app provides production authentication or a hardened public endpoint.

### Embeddings, retrieval, and multimodal search

| File | Concept | Execution notes |
| --- | --- | --- |
| [`qdrant_collection_design.py`](examples/qdrant_collection_design.py) | Collection schema, on-disk vectors, INT8 quantization, payload indices, and tenant filters. | Needs `qdrant-client` and local Qdrant; creates collections/indices without recreation. Tenant filtering is not authorization; IDs must be globally unique UUIDs and upserts can overwrite matching IDs. Reserved metadata fields are rejected. Offline checks: `python -m unittest examples.test_qdrant_collection_design -v`. |
| [`hybrid_search.py`](examples/hybrid_search.py) | Dense OpenAI embeddings + BM25 keyword ranking, combined with Reciprocal Rank Fusion. | Needs `numpy`, `openai`, `rank-bm25`, and an OpenAI key. Uses whitespace tokenization and makes paid embedding requests; RRF scores are rank-based, not probabilities. Offline checks: `python -m unittest examples.test_hybrid_search -v`. |
| [`qdrant_search.py`](examples/qdrant_search.py) | Qdrant cosine search, rich payloads, AND filters, and a keyword payload index. | Needs `qdrant-client`, `openai`, an OpenAI key, and local Qdrant on port 6333. Creates `ai_documents` if absent; upserts overwrite IDs 0–4. Use disposable data; existing schema must support 1536-D cosine vectors. Offline checks: `python -m unittest examples.test_qdrant_search -v`. |
| [`brute_force_benchmark.py`](examples/brute_force_benchmark.py) | NumPy exact search latency and vector storage at increasing scales. | NumPy only; no API calls. Largest case stores about 2.9 GiB of vectors, with higher peak memory. Timings include full sorting and depend on hardware; no ANN comparison is performed. |
| [`pgvector_search.py`](examples/pgvector_search.py) | OpenAI embeddings, pgvector HNSW cosine search, and parameterized metadata filters. | Needs `asyncpg`, `openai`, an OpenAI key/API billing, and a running PostgreSQL `ai_service` database with pgvector installed and extension/table/index creation permissions. Creates schema objects and inserts rows; reruns duplicate data, and existing schema must match. Offline tests: `python -m unittest examples.test_pgvector_search -v` (stdlib only). |
| [`similarity_metrics.py`](examples/similarity_metrics.py) | Cosine similarity, dot product, Euclidean distance, normalization, rankings. | OpenAI + NumPy. Ranking equivalence assumes unit-normalized vectors, subject to rounding/ties. |
| [`embedding_comparison.py`](examples/embedding_comparison.py) | OpenAI default/reduced dimensions vs local MiniLM. | OpenAI + NumPy + sentence-transformers. Local latency includes loading; a three-text spread is not a quality benchmark. |
| [`embedding_evaluation.py`](examples/embedding_evaluation.py) | Recall@k, MRR, NDCG across two OpenAI embedding models. | OpenAI + NumPy; tiny labeled retrieval dataset. MRR is truncated to retrieved top-k results. |
| [`multimodal_search.py`](examples/multimodal_search.py) | CLIP text/image retrieval in one model's shared space. | NumPy + sentence-transformers + Pillow; local inference, initial model download. Main demonstrates text-only search; image helpers need your own image files. |
| [`production_embedding_pipeline.py`](examples/production_embedding_pipeline.py) | JSON cache, incremental embedding, drift checks. | OpenAI + NumPy; writes `./embedding_cache/`. Drift checks make fresh paid calls. Cache writes are not atomic, concurrent writers unsupported, loaded data unvalidated. |

## Lab: embedding explorer

The lab lives in [`src/labs/embedding_explorer/`](src/labs/embedding_explorer/),
not a root `labs/` directory. It compares **12 English texts in 3 categories**:
`tech`, `cooking`, and `sports`, with four texts each.

### Configure once, then explore

Install the lab dependencies above. Create your own local
`src/labs/embedding_explorer/.env` with an `OPENAI_API_KEY` entry using a private
editor, or securely export that variable instead. The generator explicitly
loads the `.env` beside its script, **not** the root `.env`, and does not override
an existing environment variable. Never publish the real value.

Run these from the repository root with your virtual environment activated:

```bash
# 1. Inspect the corpus count (no API call)
python src/labs/embedding_explorer/corpus.py

# 2. Generate embeddings and print their shapes
python src/labs/embedding_explorer/generate_embeddings.py

# 3. Print per-category cosine similarity blocks and means
python src/labs/embedding_explorer/cosine_similarity_matrices.py

# 4. Save a non-interactive four-panel visualization
MPLBACKEND=Agg python src/labs/embedding_explorer/visualize_embeddings.py
```

The final command uses POSIX shell syntax; in PowerShell set
`$env:MPLBACKEND = "Agg"` before running the Python command.

### What each step means

- [`corpus.py`](src/labs/embedding_explorer/corpus.py) defines `TEXTS` and
  matching `LABELS`. Indices 0–3 are tech, 4–7 cooking, and 8–11 sports.
- [`generate_embeddings.py`](src/labs/embedding_explorer/generate_embeddings.py)
  calls OpenAI `text-embedding-3-small` first, then local `all-MiniLM-L6-v2`
  with normalized output. Expected default shapes are `(12, 1536)` and
  `(12, 384)` respectively. It prints labeled shapes, not vector contents.
- [`cosine_similarity_matrices.py`](src/labs/embedding_explorer/cosine_similarity_matrices.py)
  prints three intra-category and three unordered cross-category 4×4 blocks
  **for each model separately**, with original corpus indices for rows/columns.
  Cross-category means compare topics, never vectors from different models.
- Intra-category means **include diagonal self-similarities**, normally 1 for
  nonzero vectors. They therefore are not distinct-document-only averages;
  block means are rounded to four decimal places.
- [`visualize_embeddings.py`](src/labs/embedding_explorer/visualize_embeddings.py)
  independently fits t-SNE and UMAP per model. Rows represent models, columns
  reduction methods; colors are known labels, not inferred cluster assignments.
  It saves `src/labs/embedding_explorer/plots/embedding_projections.png`, prints
  the path, and closes the figure without opening an interactive window.

Each of steps 2–4 **regenerates embeddings**, so running all three makes three
OpenAI embedding requests and reloads the local model in each process. There
is no saved embedding dataset or handoff between scripts. Downloaded model
weights may be cached by the underlying library; that is not embedding persistence.

Never concatenate or compare OpenAI vectors directly with MiniLM vectors:
their dimensions and coordinate spaces differ. Likewise, 2D axes and distances
are not comparable across panels. Both reducers use cosine distance and seed
42; t-SNE perplexity and UMAP neighbors are capped at 5 (and below sample count).
With only 12 texts, projections can distort structure substantially. Seeds aid
repeatability but do not guarantee identical plots across library versions or
platforms. Apparent separation is not evidence that one model is better.

## Tests: offline checks, not provider integration tests

Run explicitly from the root after installing the relevant dependencies:

```bash
python -m unittest discover -s src/labs/embedding_explorer -p 'test_*.py' -v
python -m pytest examples/test_fastapi_app.py
```

The lab tests mock provider SDKs, embedding generation, and reducer calls as
appropriate, while checking matrix arithmetic and real headless PNG rendering.
They do not validate live model availability, billing, or real reducer quality.
NumPy, scikit-learn, and Matplotlib are needed for the complete lab test suite;
no real keys or model downloads are required by these mocked checks.

The FastAPI tests cover success, rate-limit translation, input validation, and
health responses. `pytest` without a path uses `testpaths = ["tests"]` from
`pyproject.toml`, so it does not automatically select these example/lab tests.

## Before making a public copy

Keep generated PNGs, model/tool caches, virtual environments, and the pipeline's
`embedding_cache/` out of Git. The lab plot is generated locally, not a required
source asset. Verify ignore rules for these outputs before staging; do not
assume every generated path is already excluded. `.env` is ignored, but always
inspect staged filenames and review for secrets before publishing.

Publishing source publicly does not mean exposing an unauthenticated running
API. Keep the web examples local until authentication, error sanitization,
request limits, data handling, and cost controls have been implemented.
