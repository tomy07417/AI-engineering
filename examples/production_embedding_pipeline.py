"""Demonstrate embedding caching, incremental updates, and drift monitoring.

Run: python3 examples/production_embedding_pipeline.py
Requires numpy, openai, and OPENAI_API_KEY. API requests incur costs.
Running the example persists embeddings under ./embedding_cache.
This is a learning example, not a production-hardened cache: JSON writes are
not atomic, concurrent writers are unsupported, and loaded data is not validated.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

CACHE_DIR = Path("./embedding_cache")


@dataclass
class EmbeddingRecord:
    text_hash: str
    embedding: list[float]
    model: str
    timestamp: float


class ProductionEmbeddingPipeline:
    """Embedding pipeline with caching, incremental updates, and drift detection."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.cache: dict[str, EmbeddingRecord] = {}
        self._load_cache()

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _load_cache(self):
        cache_file = CACHE_DIR / f"{self.model.replace('/', '_')}_cache.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            for h, record in data.items():
                self.cache[h] = EmbeddingRecord(**record)

    def _save_cache(self):
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{self.model.replace('/', '_')}_cache.json"
        data = {h: vars(r) for h, r in self.cache.items()}
        cache_file.write_text(json.dumps(data))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with caching — only request embeddings for uncached texts."""
        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            h = self._text_hash(text)
            if h in self.cache and self.cache[h].model == self.model:
                results[i] = self.cache[h].embedding
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        cache_hits = len(texts) - len(uncached_texts)
        print(
            f"Cache: {cache_hits}/{len(texts)} hits, "
            f"{len(uncached_texts)} texts need embedding"
        )

        if uncached_texts:
            with OpenAI() as client:
                resp = client.embeddings.create(input=uncached_texts, model=self.model)
            for j, emb_data in enumerate(resp.data):
                idx = uncached_indices[j]
                results[idx] = emb_data.embedding
                h = self._text_hash(uncached_texts[j])
                self.cache[h] = EmbeddingRecord(
                    text_hash=h,
                    embedding=emb_data.embedding,
                    model=self.model,
                    timestamp=time.time(),
                )
            self._save_cache()

        ordered = [results[i] for i in range(len(texts))]
        return np.array(ordered)

    def detect_drift(self, reference_texts: list[str], threshold: float = 0.05) -> dict:
        """Detect embedding drift by comparing cached vs fresh embeddings."""
        cached_embs = []
        for text in reference_texts:
            h = self._text_hash(text)
            if h in self.cache:
                cached_embs.append(self.cache[h].embedding)
            else:
                return {"drift_detected": False, "reason": "No cached embeddings found"}

        with OpenAI() as client:
            fresh_resp = client.embeddings.create(input=reference_texts, model=self.model)
        fresh_embs = [e.embedding for e in fresh_resp.data]

        drifts = []
        for cached, fresh in zip(cached_embs, fresh_embs):
            c, f = np.array(cached), np.array(fresh)
            sim = float(np.dot(c, f) / (np.linalg.norm(c) * np.linalg.norm(f)))
            drifts.append(1.0 - sim)

        avg_drift = float(np.mean(drifts))
        return {
            "drift_detected": avg_drift > threshold,
            "avg_drift": round(avg_drift, 6),
            "max_drift": round(float(np.max(drifts)), 6),
            "threshold": threshold,
        }


def main() -> None:
    """Demonstrate cold and warm caches, a new document, and drift detection."""
    pipeline = ProductionEmbeddingPipeline()
    texts = ["RAG pipeline architecture", "Vector database comparison", "Embedding models overview"]

    # First call requests embeddings unless a previous run populated the cache.
    pipeline.embed(texts)

    # Second call reads all embeddings from the cache.
    pipeline.embed(texts)

    # Incremental: add new text.
    all_texts = texts + ["New document about fine-tuning"]
    pipeline.embed(all_texts)

    # Check drift with a fresh API request, without replacing the cached embeddings.
    drift = pipeline.detect_drift(texts[:2])
    print(f"Drift: {drift}")


if __name__ == "__main__":
    main()
