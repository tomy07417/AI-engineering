"""Implement and compare cosine similarity, dot product, and Euclidean distance.

Run: python3 examples/similarity_metrics.py
Requires numpy, openai, and OPENAI_API_KEY in the environment.
Running this example makes real API requests and incurs API costs.
"""

import numpy as np
from openai import OpenAI


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def get_embeddings(texts: list[str]) -> np.ndarray:
    with OpenAI() as client:
        resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([e.embedding for e in resp.data])


def compare_metrics(texts: list[str]) -> None:
    """Compare all three metrics on a set of texts."""
    embeddings = get_embeddings(texts)

    print(f"{'Pair':<35} {'Cosine':>8} {'Dot':>8} {'L2 Dist':>8}")
    print("-" * 65)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            label = f"[{i}] vs [{j}]"
            cos = cosine_similarity(embeddings[i], embeddings[j])
            dot = dot_product(embeddings[i], embeddings[j])
            euc = euclidean_distance(embeddings[i], embeddings[j])
            print(f"{label:<35} {cos:>8.4f} {dot:>8.4f} {euc:>8.4f}")


def verify_normalization(texts: list[str]) -> None:
    """Verify that OpenAI embeddings are normalized (L2 norm ≈ 1)."""
    embeddings = get_embeddings(texts)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 norms: {norms.round(6)}")
    print(f"All normalized (norm ≈ 1): {np.allclose(norms, 1.0, atol=1e-5)}")


def rank_by_all_metrics(query: str, corpus: list[str]) -> None:
    """Show equivalent rankings for unit-normalized embeddings, up to rounding and ties."""
    all_texts = [query] + corpus
    embeddings = get_embeddings(all_texts)
    q_emb = embeddings[0]
    doc_embs = embeddings[1:]

    for metric_name, metric_fn, reverse in [
        ("Cosine", cosine_similarity, True),
        ("Dot Product", dot_product, True),
        ("Euclidean", euclidean_distance, False),
    ]:
        scores = [(corpus[i], metric_fn(q_emb, doc_embs[i])) for i in range(len(corpus))]
        scores.sort(key=lambda x: x[1], reverse=reverse)
        ranking = [s[0][:40] for s in scores]
        print(f"{metric_name:>12}: {' > '.join(ranking)}")


def main() -> None:
    """Compare metrics, check normalization, and display retrieval rankings."""
    texts = [
        "Python is a programming language",
        "Python is used for AI development",
        "The weather is sunny today",
    ]

    compare_metrics(texts)
    verify_normalization(texts)

    print("\nRankings for query: 'AI and machine learning'")
    rank_by_all_metrics("AI and machine learning", texts)


if __name__ == "__main__":
    main()
