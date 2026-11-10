"""Offline checks: python -m unittest examples.test_hybrid_search -v."""

import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

SCRIPT = Path(__file__).with_name("hybrid_search.py")


class HybridSearchTests(unittest.TestCase):
    def load_example(self):
        openai, bm25 = MagicMock(), MagicMock()
        with patch.dict("sys.modules", {"openai": openai, "rank_bm25": bm25}):
            module = runpy.run_path(str(SCRIPT))
        openai.OpenAI.assert_not_called()
        bm25.BM25Okapi.assert_not_called()
        return module, bm25

    def test_rrf_combines_rank_positions_not_raw_scores(self):
        module, _ = self.load_example()
        fused = module["rrf_fusion"]([[0, 1], [1, 2]], k=60)
        self.assertEqual([doc for doc, _ in fused], [1, 0, 2])
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused[1][1], 1 / 61)
        self.assertAlmostEqual(fused[2][1], 1 / 62)
        self.assertEqual(module["rrf_fusion"]([]), [])

    def test_dense_sparse_tokenization_and_hybrid_results(self):
        module, bm25 = self.load_example()
        cls = module["HybridSearchEngine"]
        docs = ["PGVECTOR HNSW", "BM25 Search", "Vector Database"]
        embed = MagicMock(side_effect=[np.eye(3), np.array([[0., 1., 0.]])])
        with patch.dict(cls.__init__.__globals__, embed=embed):
            engine = cls(docs)
            self.assertEqual(engine.search_dense("question", top_k=1), [1])
        bm25.BM25Okapi.assert_called_once_with(
            [["pgvector", "hnsw"], ["bm25", "search"], ["vector", "database"]]
        )
        engine.bm25.get_scores.return_value = np.array([2., 0., 1.])
        self.assertEqual(engine.search_sparse("PGVECTOR HNSW", 2), [0, 2])
        engine.bm25.get_scores.assert_called_once_with(["pgvector", "hnsw"])
        with patch.object(engine, "search_dense", return_value=[0, 1]) as dense:
            with patch.object(engine, "search_sparse", return_value=[1, 2]) as sparse:
                results = engine.search_hybrid("query", top_k=3)
        dense.assert_called_once_with("query", top_k=20)
        sparse.assert_called_once_with("query", top_k=20)
        self.assertEqual(results, [
            {"text": docs[1], "rrf_score": round(1 / 62 + 1 / 61, 4),
             "dense_rank": 2, "sparse_rank": 1},
            {"text": docs[0], "rrf_score": round(1 / 61, 4),
             "dense_rank": 1, "sparse_rank": None},
            {"text": docs[2], "rrf_score": round(1 / 62, 4),
             "dense_rank": None, "sparse_rank": 2},
        ])


if __name__ == "__main__":
    unittest.main()
