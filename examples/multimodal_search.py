"""Multimodal embeddings with CLIP — search images with text and vice versa.

Run: python3 examples/multimodal_search.py
Requires numpy, Pillow, and sentence-transformers. The model downloads on first use.
Inference runs locally without an API key.
"""

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearchEngine:
    """Search across text and images using CLIP embeddings."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.items: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def add_texts(self, texts: list[str]):
        """Add text documents to the index."""
        embs = self.model.encode(texts, normalize_embeddings=True)
        for text, emb in zip(texts, embs):
            self.items.append({"type": "text", "content": text})
        self._update_embeddings(embs)

    def add_images(self, image_paths: list[str]):
        """Add images to the index."""
        images = [Image.open(p) for p in image_paths]
        embs = self.model.encode(images, normalize_embeddings=True)
        for path, emb in zip(image_paths, embs):
            self.items.append({"type": "image", "content": path})
        self._update_embeddings(embs)

    def _update_embeddings(self, new_embs: np.ndarray):
        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

    def search_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """Search all items (text + images) using a text query."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        return self._search(q_emb, top_k)

    def search_by_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        """Search all items using an image query."""
        img = Image.open(image_path)
        q_emb = self.model.encode([img], normalize_embeddings=True)[0]
        return self._search(q_emb, top_k)

    def _search(self, query_emb: np.ndarray, top_k: int) -> list[dict]:
        if self.embeddings is None:
            return []
        scores = self.embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{**self.items[i], "score": float(scores[i])} for i in top_indices]


def main() -> None:
    """Demonstrate text-to-text search using CLIP."""
    engine = MultimodalSearchEngine()

    engine.add_texts([
        "A golden retriever playing in the park",
        "Python programming language documentation",
        "Neural network architecture diagram",
        "Sunset over the ocean with palm trees",
    ])

    results = engine.search_by_text("machine learning models")
    for r in results[:3]:
        print(f"  [{r['score']:.3f}] [{r['type']}] {r['content']}")


if __name__ == "__main__":
    main()
