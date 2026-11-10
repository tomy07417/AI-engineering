"""Illustrative PostgreSQL vector search with OpenAI embeddings.

Run from the repository root: python examples/pgvector_search.py
Prerequisites: asyncpg and openai Python packages; a running PostgreSQL server
with pgvector installed (HNSW support), an existing ai_service database at
postgresql://localhost/ai_service, and permissions to create the vector
extension, table, and index and to insert/query rows. Supply OPENAI_API_KEY
through your environment; requests use real OpenAI API billing.

Running creates the extension, documents table, and HNSW index if absent and
inserts three documents. Reruns insert duplicate rows. An existing documents
table must match this schema; CREATE IF NOT EXISTS does not migrate it.
Synchronous embedding calls block the event loop: this is illustrative code,
not a production service. Importing does not create clients or make requests.
"""

from __future__ import annotations

import asyncio
import json

import asyncpg
from openai import OpenAI


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with the model's default 1536 dimensions."""
    client = OpenAI()
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [item.embedding for item in response.data]


async def setup_database(conn: asyncpg.Connection) -> None:
    """Create the demonstration schema and cosine-distance HNSW index."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)


async def insert_documents(conn: asyncpg.Connection, documents: list[dict]) -> None:
    """Embed and insert documents, passing JSON and vectors as text for casts."""
    embeddings = embed([doc["content"] for doc in documents])
    for doc, embedding in zip(documents, embeddings):
        await conn.execute(
            """INSERT INTO documents (content, metadata, embedding)
               VALUES ($1, $2::jsonb, $3::vector)""",
            doc["content"], json.dumps(doc["metadata"]), str(embedding),
        )


async def semantic_search(
    conn: asyncpg.Connection,
    query: str,
    top_k: int = 5,
    metadata_filter: dict[str, str] | None = None,
) -> list[dict]:
    """Return cosine similarity matches, optionally matching metadata as text."""
    params = [str(embed([query])[0])]
    conditions = []
    for key, value in (metadata_filter or {}).items():
        key_position = len(params) + 1
        params.extend([key, value])
        conditions.append(
            f"metadata->> ${key_position}::text = ${key_position + 1}::text"
        )
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(top_k)
    rows = await conn.fetch(
        "SELECT content, metadata, 1 - (embedding <=> $1::vector) AS similarity "
        "FROM documents" + where
        + f" ORDER BY embedding <=> $1::vector LIMIT ${len(params)}",
        *params,
    )
    return [
        {
            "content": row["content"],
            "metadata": json.loads(row["metadata"]),
            "similarity": round(row["similarity"], 4),
        }
        for row in rows
    ]


async def main() -> None:
    conn = await asyncpg.connect("postgresql://localhost/ai_service")
    try:
        await setup_database(conn)
        await insert_documents(conn, [
            {"content": "RAG combines retrieval with language model generation.",
             "metadata": {"topic": "rag"}},
            {"content": "PostgreSQL with pgvector stores and searches vector embeddings.",
             "metadata": {"topic": "database"}},
            {"content": "HNSW algorithms enable approximate nearest neighbor search.",
             "metadata": {"topic": "algorithms"}},
        ])
        results = await semantic_search(conn, "vector search technology", top_k=2)
        print("Semantic search:", results)
        filtered = await semantic_search(
            conn, "algorithms", top_k=2, metadata_filter={"topic": "algorithms"}
        )
        print("Filtered search:", filtered)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
