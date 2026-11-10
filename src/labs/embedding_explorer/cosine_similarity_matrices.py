"""Inspect intra-category and cross-category cosine similarity per model.

From the repository root (using the lab's existing environment):
    src/labs/embedding_explorer/.venv/bin/python src/labs/embedding_explorer/cosine_similarity_matrices.py
From the lab directory:
    .venv/bin/python cosine_similarity_matrices.py

Running this script calls generate_embeddings: OpenAI sends the corpus to its
API and incurs usage costs; SentenceTransformer may download all-MiniLM-L6-v2
on its first run. Importing this module does not generate embeddings.

For each model, print three intra-category 4x4 matrices and three cross-category
4x4 matrices (one per unordered pair). Categories follow their first occurrence
in LABELS; documents within each category retain corpus order. Row/column keys
use original corpus indices: 0-3:tech, 4-7:cooking, 8-11:sports. Cross-category
blocks compare the row category against the column category, not two models.
Scores describe document similarity within each model's own embedding space;
they are not clustering assignments or cross-model comparisons.

Each block also prints its mean, rounded to four decimal places. Intra-category
means include the diagonal, exactly like block.mean(): self comparisons are 1
for nonzero embeddings, so including them raises the mean relative to comparing
only distinct documents. No diagonal entries are excluded.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from corpus import LABELS
from generate_embeddings import generate_embeddings


def build_category_matrices(embeddings: np.ndarray, labels: list[str]) -> dict:
    """Return intra and unordered cross blocks in first-occurrence label order."""
    similarities = cosine_similarity(embeddings)
    indices = {
        category: [index for index, label in enumerate(labels) if label == category]
        for category in dict.fromkeys(labels)
    }
    categories = list(indices)
    intra = {
        category: similarities[np.ix_(members, members)]
        for category, members in indices.items()
    }
    cross = {
        (left, right): similarities[np.ix_(indices[left], indices[right])]
        for index, left in enumerate(categories)
        for right in categories[index + 1:]
    }
    return {"intra": intra, "cross": cross}


def main() -> None:
    """Generate once and print category blocks independently for each model."""
    openai_embeddings, local_embeddings = generate_embeddings()
    keys = {
        category: " ".join(
            f"{index}:{label}" for index, label in enumerate(LABELS)
            if label == category
        ) for category in dict.fromkeys(LABELS)
    }
    with np.printoptions(precision=3, suppress=True):
        for model, embeddings in (
            ("OpenAI", openai_embeddings), ("SentenceTransformer", local_embeddings)
        ):
            blocks = build_category_matrices(embeddings, LABELS)
            print(f"{model} cosine similarity")
            for category, matrix in blocks["intra"].items():
                print(f"Intra-category: {category}")
                print(f"Rows: {keys[category]}")
                print(f"Columns: {keys[category]}")
                print(matrix)
                print(f"Mean: {round(float(matrix.mean()), 4)}")
            for (left, right), matrix in blocks["cross"].items():
                print(f"Cross-category: {left} vs {right}")
                print(f"Rows: {keys[left]}")
                print(f"Columns: {keys[right]}")
                print(matrix)
                print(f"Mean: {round(float(matrix.mean()), 4)}")


if __name__ == "__main__":
    main()
