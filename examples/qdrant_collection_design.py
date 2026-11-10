"""Collection design patterns: multi-tenant payload filtering with Qdrant.

Run: python examples/qdrant_collection_design.py
Requires qdrant-client and a local Qdrant server on port 6333. No embedding API
is called: upsert/search helpers expect caller-provided vectors.
Running creates knowledge_base if absent and creates payload indices. Existing
collections are never recreated; their schema must already match this example.

Tenant filters are NOT authorization. Obtain tenant_id from authenticated server
context, enforce it on every operation, and do not expose a raw Qdrant client to
untrusted users. Point IDs are collection-wide: use globally unique valid UUID
strings for doc_id, not tenant-local IDs; upsert replaces an existing matching ID
regardless of tenant. This example does not implement ownership checks or auth.
Metadata may not override reserved tenant_id/content fields. Datetime payloads
must use Qdrant-supported timestamps; equality filters shown here are basic,
not datetime range filters. Quantized INT8 copies stay in RAM, original vectors
on disk and the HNSW graph in RAM: total memory includes all these structures.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient, models


@dataclass
class CollectionSchema:
    name: str
    vector_size: int
    distance: str
    payload_indices: list[str]
    description: str


SCHEMAS = [
    CollectionSchema(
        name="knowledge_base",
        vector_size=1536,
        distance="Cosine",
        payload_indices=["tenant_id", "doc_type", "language", "created_at"],
        description="Main multi-tenant knowledge base",
    ),
]


def create_collection(client: QdrantClient, schema: CollectionSchema):
    """Create a missing collection and ensure the requested payload indices."""
    if not client.collection_exists(collection_name=schema.name):
        client.create_collection(
            collection_name=schema.name,
            vectors_config=models.VectorParams(
                size=schema.vector_size,
                distance=models.Distance(schema.distance),
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200, on_disk=False),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=True,
                ),
            ),
        )

    for field in schema.payload_indices:
        field_type = (
            models.PayloadSchemaType.DATETIME
            if field == "created_at"
            else models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=schema.name,
            field_name=field,
            field_schema=field_type,
            wait=True,
        )


def upsert_document(
    client: QdrantClient,
    collection: str,
    doc_id: str,
    embedding: list[float],
    tenant_id: str,
    content: str,
    metadata: dict,
):
    """Upsert a globally unique UUID point with a trusted tenant payload."""
    if {"tenant_id", "content"} & metadata.keys():
        raise ValueError("metadata contains reserved fields: tenant_id or content")
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"tenant_id": tenant_id, "content": content, **metadata},
        )],
        wait=True,
    )


def search_tenant(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    tenant_id: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Apply the trusted tenant condition AND any additional equality filters."""
    must_conditions = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
    ]
    if filters:
        for key, value in filters.items():
            must_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=models.Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "content": point.payload.get("content", ""),
            "metadata": {
                key: value for key, value in point.payload.items()
                if key not in ("content", "tenant_id")
            },
            "score": point.score,
        }
        for point in results.points
    ]


def main() -> None:
    """Set up example collections without deleting existing data."""
    client = QdrantClient(url="http://localhost:6333")
    try:
        for schema in SCHEMAS:
            create_collection(client, schema)
            print(f"Collection ready: {schema.name} ({schema.description})")
    finally:
        client.close()


if __name__ == "__main__":
    main()
