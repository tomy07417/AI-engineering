"""Compare OpenAI and local sentence-transformers embedding models.

Run: python3 examples/embedding_comparison.py
Requires numpy, openai, sentence-transformers, and OPENAI_API_KEY.
OpenAI requests incur API costs. The local model may download on first use.
Local latency includes model loading; spread is illustrative, not a quality benchmark.
"""

import time

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def embed_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int | None = None,
) -> np.ndarray:
    """OpenAI embeddings with optional dimension reduction."""
    kwargs = {"input": texts, "model": model}
    if dimensions:
        kwargs["dimensions"] = dimensions
    with OpenAI() as client:
        resp = client.embeddings.create(**kwargs)
    return np.array([e.embedding for e in resp.data])


def embed_local(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Local sentence-transformers embedding (free, no API key)."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True)


def benchmark_model(name: str, embed_fn, texts: list[str]) -> dict:
    """Benchmark latency and illustrative similarity separation."""
    start = time.perf_counter()
    embeddings = embed_fn(texts)
    elapsed = time.perf_counter() - start

    # Cosine similarity between first two (should be similar).
    sim_similar = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
    # Cosine similarity between first and last (should be different).
    sim_different = float(
        np.dot(embeddings[0], embeddings[-1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[-1]))
    )

    return {
        "model": name,
        "dimensions": embeddings.shape[1],
        "latency_ms": round(elapsed * 1000),
        "sim_similar": round(sim_similar, 3),
        "sim_different": round(sim_different, 3),
        "spread": round(sim_similar - sim_different, 3),
    }


def main() -> None:
    """Compare models on two similar texts and one unrelated text."""
    test_texts = [
        "How to implement RAG with Python and OpenAI",
        "Building a retrieval augmented generation system in Python",
        "Recipe for homemade chocolate chip cookies with butter",
    ]

    results = [
        benchmark_model("openai-small-1536", lambda t: embed_openai(t), test_texts),
        benchmark_model(
            "openai-small-256", lambda t: embed_openai(t, dimensions=256), test_texts
        ),
        benchmark_model("MiniLM-L6 (local)", lambda t: embed_local(t), test_texts),
    ]

    print(
        f"{'Model':<25} {'Dims':>5} {'Latency':>8} "
        f"{'Similar':>8} {'Different':>10} {'Spread':>7}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['model']:<25} {r['dimensions']:>5} {r['latency_ms']:>6}ms "
            f"{r['sim_similar']:>8.3f} {r['sim_different']:>10.3f} {r['spread']:>7.3f}"
        )


if __name__ == "__main__":
    main()
