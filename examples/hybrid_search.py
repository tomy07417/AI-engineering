"""Hybrid search: dense embeddings + BM25 keyword search with RRF fusion.

Run: python examples/hybrid_search.py
Requires numpy, rank-bm25, openai, and OPENAI_API_KEY in the environment.
Running this example sends texts to OpenAI and incurs API costs; import does not.
Dense scoring assumes unit-normalized OpenAI embeddings. BM25 uses simple
lowercase/whitespace tokenization. RRF combines rank positions, not raw scores;
its score is not a probability or cosine similarity. Each dense search makes
a fresh query embedding request, including the dense step of hybrid search.
"""

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi


def embed(texts: list[str]) -> np.ndarray:
    with OpenAI() as client:
        resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([e.embedding for e in resp.data])


def rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion — merge multiple ranked lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearchEngine:
    """Combine dense (embedding) and sparse (BM25) search."""

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.embeddings = embed(documents)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search_dense(self, query: str, top_k: int = 10) -> list[int]:
        q_emb = embed([query])[0]
        scores = self.embeddings @ q_emb
        return list(np.argsort(scores)[::-1][:top_k])

    def search_sparse(self, query: str, top_k: int = 10) -> list[int]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        return list(np.argsort(scores)[::-1][:top_k])

    def search_hybrid(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        dense_ranking = self.search_dense(query, top_k=20)
        sparse_ranking = self.search_sparse(query, top_k=20)
        fused = rrf_fusion([dense_ranking, sparse_ranking], k=rrf_k)

        results = []
        for doc_id, score in fused[:top_k]:
            results.append({
                "text": self.documents[doc_id],
                "rrf_score": round(score, 4),
                "dense_rank": dense_ranking.index(doc_id) + 1 if doc_id in dense_ranking else None,
                "sparse_rank": sparse_ranking.index(doc_id) + 1 if doc_id in sparse_ranking else None,
            })
        return results


def main() -> None:
    """Show dense, sparse, and fused rankings for the same query."""
    corpus = [
        "HNSW algorithm provides efficient approximate nearest neighbor search",
        "PostgreSQL with pgvector extension enables vector similarity queries",
        "BM25 is a classic information retrieval scoring function",
        "Qdrant vector database written in Rust for high performance",
        "ChromaDB API reference documentation version 0.5.0",
        "Reciprocal rank fusion combines multiple search result lists",
        "The pgvector extension supports IVFFlat and HNSW index types",
    ]
    engine = HybridSearchEngine(corpus)
    query = "pgvector HNSW index"
    print(f"Query: '{query}'\n")

    print("Dense (semantic) results:")
    for idx in engine.search_dense(query, 3):
        print(f"  {corpus[idx]}")

    print("\nSparse (BM25) results:")
    for idx in engine.search_sparse(query, 3):
        print(f"  {corpus[idx]}")

    print("\nHybrid (RRF fusion) results:")
    for result in engine.search_hybrid(query, top_k=3):
        print(
            f"  [RRF:{result['rrf_score']:.4f}] dense:{result['dense_rank']} "
            f"sparse:{result['sparse_rank']} — {result['text']}"
        )


if __name__ == "__main__":
    main()
