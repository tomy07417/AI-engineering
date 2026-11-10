"""Compare category-colored t-SNE and UMAP projections for two embedding models.

Run from the repository root with the existing lab environment:
    src/labs/embedding_explorer/.venv/bin/python src/labs/embedding_explorer/visualize_embeddings.py

Dependencies: numpy, matplotlib, scikit-learn, umap-learn, plus the existing
embedding generator's openai, sentence-transformers and python-dotenv.
Running calls generate_embeddings once: OpenAI sends the corpus to its API
and incurs usage costs; SentenceTransformer may download all-MiniLM-L6-v2
on its first run. Importing this module does not call the generator or save files.

Saves plots/embedding_projections.png beside this script and prints its path.
No interactive window is opened. Rows are models; columns are reduction methods.
Colors indicate known corpus categories, not inferred clusters. Each panel is
fit independently: axes and distances are not comparable across panels. These
stochastic 2D views distort high-dimensional structure, especially for only 12
texts; apparent separation is not evidence of embedding quality.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

from corpus import LABELS
from generate_embeddings import generate_embeddings


def reduce_embeddings(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    """Fit two independent, seeded 2D cosine projections without changing input."""
    n_samples = len(embeddings)
    if n_samples < 3:
        raise ValueError("At least 3 texts are required for t-SNE and UMAP.")
    return {
        "tsne": TSNE(
            n_components=2, metric="cosine", init="random", learning_rate="auto",
            perplexity=min(5, n_samples - 1), random_state=42,
        ).fit_transform(embeddings),
        "umap": UMAP(
            n_components=2, metric="cosine", init="random",
            n_neighbors=min(5, n_samples - 1), random_state=42, n_jobs=1,
        ).fit_transform(embeddings),
    }


def plot_projections(
    projections: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    output_path: Path,
) -> None:
    """Save a model-by-method grid with stable category colors, then close it."""
    categories = list(dict.fromkeys(labels))
    colors = {category: plt.get_cmap("tab10")(i) for i, category in enumerate(categories)}
    figure, axes = plt.subplots(len(projections), 2, figsize=(12, 9), squeeze=False)
    try:
        for row, (model, coordinates) in enumerate(projections.items()):
            for column, (method, name) in enumerate((("tsne", "t-SNE"), ("umap", "UMAP"))):
                axis = axes[row, column]
                for category in categories:
                    indices = [i for i, label in enumerate(labels) if label == category]
                    points = coordinates[method][indices]
                    axis.scatter(points[:, 0], points[:, 1], color=colors[category], label=category)
                axis.set_title(f"{model} — {name}")
                axis.set_xlabel("Component 1")
                axis.set_ylabel("Component 2")
                axis.legend(title="Category")
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)


def main() -> None:
    """Generate once, reduce each model separately, and save the four panels."""
    openai_embeddings, local_embeddings = generate_embeddings()
    projections = {
        "OpenAI": reduce_embeddings(openai_embeddings),
        "SentenceTransformer": reduce_embeddings(local_embeddings),
    }
    output_path = Path(__file__).resolve().parent / "plots" / "embedding_projections.png"
    plot_projections(projections, LABELS, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
