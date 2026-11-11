"""Four composable ingestion functions; importing does not load a model or connect."""

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


UPSERT_SQL = """
    INSERT INTO documents (title, content, embedding, metadata)
    VALUES ($1, $2, $3::vector, $4::jsonb)
    ON CONFLICT (title) DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
"""


def _validated_document(document: dict) -> dict:
    if not isinstance(document, dict):
        raise ValueError("Each document must be a dictionary.")
    for field in ("title", "content"):
        value = document.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Document {field} must be a nonblank string.")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Document metadata must be a JSON object.")
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Document metadata must contain valid JSON values.") from exc
    return {"title": document["title"], "content": document["content"], "metadata": metadata}


def _checked_vectors(values: Any, count: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (count, 384):
        raise ValueError(f"Expected {count} embeddings with 384 dimensions.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain only finite values.")
    return vectors


def load_document(path: str | Path, *, root: str | Path | None = None) -> dict:
    """Read one UTF-8 .txt/.md file; its relative path is its database identity."""
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
        raise ValueError("Document must be an existing .txt or .md file.")
    base = Path(root) if root is not None else path.parent
    try:
        title = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Document must be inside the source root.") from exc
    return _validated_document({
        "title": title,
        "content": path.read_text(encoding="utf-8"),
        "metadata": {"source": title},
    })


def generate_embedding(
    texts: list[str], *, model: Any = None, cache_folder: str | None = None,
) -> np.ndarray:
    """Encode a batch as normalized 384-D vectors; optionally reuse a caller's model."""
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Embedding inputs must be nonblank strings.")
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, cache_folder=cache_folder)
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return _checked_vectors(vectors, len(texts))


async def upsert_document(connection: Any, document: dict, embedding: Any) -> None:
    """Parameterize an insert/update by title; caller owns connection/transaction.

    Existing id and created_at are preserved. Metadata is replaced, not merged.
    Vector text serialization avoids requiring connection-specific codecs.
    """
    document = _validated_document(document)
    vector = _checked_vectors([embedding], 1)[0]
    await connection.execute(
        UPSERT_SQL,
        document["title"], document["content"], json.dumps(vector.tolist(), allow_nan=False),
        json.dumps(document["metadata"], allow_nan=False),
    )


async def ingest(
    documents: list[dict], connection: Any, *, model: Any = None,
    cache_folder: str | None = None,
) -> dict[str, int]:
    """Atomically ingest a validated batch, embedding only new/changed content.

    No files are deleted. Single-writer semantics: counters describe the initial
    snapshot, not concurrent writers. Use one consistent model for all stored
    vectors; model changes require a separately planned full re-embedding.
    """
    documents = [_validated_document(document) for document in documents]
    titles = [document["title"] for document in documents]
    if len(set(titles)) != len(titles):
        raise ValueError("Document titles must be unique within an ingestion batch.")
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "embedded": 0}
    if not documents:
        return stats

    async with connection.transaction():
        rows = await connection.fetch(
            "SELECT title, content, metadata, embedding::text AS embedding "
            "FROM documents WHERE title = ANY($1::text[])", titles,
        )
        existing = {row["title"]: row for row in rows}
        changes = []
        to_embed = []
        for document in documents:
            old = existing.get(document["title"])
            needs_embedding = old is None or old["content"] != document["content"] or old["embedding"] is None
            old_metadata = old["metadata"] if old is not None else None
            if isinstance(old_metadata, str):
                old_metadata = json.loads(old_metadata)
            if not needs_embedding and old_metadata == document["metadata"]:
                stats["unchanged"] += 1
                continue
            changes.append((document, old, needs_embedding))
            if needs_embedding:
                to_embed.append(document["content"])

        vectors = (
            _checked_vectors(await asyncio.to_thread(
                generate_embedding, to_embed, model=model, cache_folder=cache_folder,
            ), len(to_embed))
            if to_embed else np.empty((0, 384), dtype=np.float32)
        )
        stats["embedded"] = len(to_embed)
        vector_index = 0
        for document, old, needs_embedding in changes:
            if needs_embedding:
                vector = vectors[vector_index]
                vector_index += 1
            else:
                vector = json.loads(old["embedding"])
            await upsert_document(connection, document, vector)
            stats["inserted" if old is None else "updated"] += 1
    return stats
