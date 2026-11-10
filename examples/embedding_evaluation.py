"""Embedding evaluation: Recall@k, MRR, NDCG, and model comparison.

Run: python3 examples/embedding_evaluation.py
Requires numpy, openai, and OPENAI_API_KEY in the environment.
Running this example makes real API requests and incurs API costs.
The reported MRR uses only the top-k retrieved documents (truncated MRR).
"""

from dataclasses import dataclass

import numpy as np
from openai import OpenAI


@dataclass
class EvalCase:
    query: str
    relevant_ids: list[int]  # Indices of relevant docs in corpus.


def embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    with OpenAI() as client:
        resp = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in resp.data])


def retrieve_top_k(query_emb: np.ndarray, corpus_embs: np.ndarray, k: int) -> list[int]:
    scores = corpus_embs @ query_emb
    return list(np.argsort(scores)[::-1][:k])


def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def mrr(retrieved: list[int], relevant: list[int]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant_set
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_model(
    model: str,
    corpus: list[str],
    eval_cases: list[EvalCase],
    k: int = 5,
) -> dict:
    """Evaluate an embedding model on a retrieval task."""
    corpus_embs = embed(corpus, model)
    recalls, mrrs, ndcgs = [], [], []

    for case in eval_cases:
        q_emb = embed([case.query], model)[0]
        retrieved = retrieve_top_k(q_emb, corpus_embs, k)
        recalls.append(recall_at_k(retrieved, case.relevant_ids, k))
        mrrs.append(mrr(retrieved, case.relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved, case.relevant_ids, k))

    return {
        "model": model,
        f"recall@{k}": round(np.mean(recalls), 3),
        "mrr": round(np.mean(mrrs), 3),
        f"ndcg@{k}": round(np.mean(ndcgs), 3),
    }


def main() -> None:
    """Compare two OpenAI embedding models on a small labeled corpus."""
    corpus = [
        "RAG combines retrieval with generation",
        "Vector databases store embeddings",
        "Fine-tuning adapts pre-trained models",
        "Prompt engineering designs effective inputs",
        "Embeddings capture semantic meaning",
        "HNSW enables approximate nearest neighbor search",
        "Chunking divides documents for retrieval",
        "LLMs generate text based on patterns",
    ]

    eval_set = [
        EvalCase("How do embeddings work?", [4, 1]),
        EvalCase("What is RAG?", [0, 6]),
        EvalCase("How to search similar vectors fast?", [5, 1]),
    ]

    for model in ["text-embedding-3-small", "text-embedding-3-large"]:
        results = evaluate_model(model, corpus, eval_set, k=3)
        print(results)


if __name__ == "__main__":
    main()
