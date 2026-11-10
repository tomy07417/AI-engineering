"""Why vector databases — benchmark NumPy brute-force search at different scales.

Run: python examples/brute_force_benchmark.py
Requires numpy; no API key or network access is needed.
The largest case stores about 2.9 GiB of vectors; peak memory is higher because
random generation starts as float64 before conversion and normalization.
Reported memory covers only the final vector matrix, not total process memory.
Timings include scoring and full sorting, but exclude vector creation.
Results depend on hardware, BLAS threading, and load; this is not an ANN benchmark.
"""

import time

import numpy as np


def benchmark_brute_force(n_vectors: int, dims: int = 1536, n_queries: int = 10) -> dict:
    """Benchmark brute-force search at different scales."""
    vectors = np.random.randn(n_vectors, dims).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    queries = np.random.randn(n_queries, dims).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    memory_mb = vectors.nbytes / 1024 / 1024

    start = time.perf_counter()
    for q in queries:
        scores = vectors @ q
        _ = np.argsort(scores)[-5:]
    elapsed = (time.perf_counter() - start) / n_queries

    return {
        "n_vectors": n_vectors,
        "memory_mb": round(memory_mb, 1),
        "avg_query_ms": round(elapsed * 1000, 1),
    }


def main() -> None:
    """Print memory and average query time for increasing corpus sizes."""
    print(f"{'Vectors':>12} {'Memory':>10} {'Query Time':>12}")
    print("-" * 38)

    for n in [1_000, 10_000, 100_000, 500_000]:
        result = benchmark_brute_force(n)
        print(
            f"{result['n_vectors']:>12,} "
            f"{result['memory_mb']:>8.1f} MB "
            f"{result['avg_query_ms']:>9.1f} ms"
        )


# At 1M vectors, float32 storage alone needs 6.144 GB (about 5.72 GiB).
# Search latency depends on hardware and workload, not a universal scale cutoff.
# ANN indices trade exactness for speed; measure recall and latency for your data.
if __name__ == "__main__":
    main()
