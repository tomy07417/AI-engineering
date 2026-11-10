"""Qdrant: vector search with rich payloads and AND filtering.

Run: python examples/qdrant_search.py
Requires qdrant-client, openai, OPENAI_API_KEY, and Qdrant at localhost:6333.
OpenAI requests send document/query text to the API and incur costs.
Use a disposable local collection: this creates ai_documents if absent and
upserts point IDs 0-4, replacing those IDs if they already exist. It never
recreates/deletes a collection. An existing collection must support 1536-D
cosine vectors. The topic keyword index is created before the filtered query.
"""

from openai import OpenAI
from qdrant_client import QdrantClient, models

COLLECTION = "ai_documents"

DOCS = [
    {"text": "RAG pipelines combine retrieval with LLM generation",
     "topic": "rag", "year": 2024, "difficulty": 3},
    {"text": "Qdrant supports advanced filtering and hybrid search",
     "topic": "vectordb", "year": 2024, "difficulty": 2},
    {"text": "HNSW provides logarithmic search complexity",
     "topic": "algorithms", "year": 2018, "difficulty": 4},
    {"text": "ChromaDB is great for local development",
     "topic": "vectordb", "year": 2023, "difficulty": 1},
    {"text": "Semantic search finds documents by meaning",
     "topic": "search", "year": 2023, "difficulty": 2},
]


def embed(texts: list[str]) -> list[list[float]]:
    """Generate OpenAI embeddings without keeping a global API client."""
    with OpenAI() as client:
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [item.embedding for item in response.data]


def main() -> None:
    """Populate a local collection and demonstrate basic and filtered search."""
    client = QdrantClient(url="http://localhost:6333")
    try:
        if not client.collection_exists(collection_name=COLLECTION):
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
            )

        embeddings = embed([doc["text"] for doc in DOCS])
        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(id=i, vector=embeddings[i], payload=doc)
                for i, doc in enumerate(DOCS)
            ],
            wait=True,
        )

        results = client.query_points(
            collection_name=COLLECTION,
            query=embed(["How do vector databases work?"])[0],
            limit=3,
            with_payload=True,
        )
        print("Semantic search:")
        for point in results.points:
            print(f"  [{point.score:.3f}] {point.payload['text']}")

        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="topic",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        filtered = client.query_points(
            collection_name=COLLECTION,
            query=embed(["database technology"])[0],
            query_filter=models.Filter(must=[
                models.FieldCondition(key="topic", match=models.MatchValue(value="vectordb")),
                models.FieldCondition(key="year", range=models.Range(gte=2024)),
            ]),
            limit=3,
            with_payload=True,
        )
        print("\nFiltered (vectordb + year>=2024):")
        for point in filtered.points:
            print(f"  [{point.score:.3f}] {point.payload['text']}")

        print(f"\nCollection info: {client.get_collection(COLLECTION).points_count} points")
    finally:
        client.close()


if __name__ == "__main__":
    main()
