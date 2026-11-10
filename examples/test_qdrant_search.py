"""Offline checks: python -m unittest examples.test_qdrant_search -v."""

import contextlib
import io
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).with_name("qdrant_search.py")


class QdrantSearchTests(unittest.TestCase):
    def load_example(self):
        openai = MagicMock()
        qdrant = MagicMock()
        for name in (
            "VectorParams", "HnswConfigDiff", "OptimizersConfigDiff", "PointStruct",
            "Filter", "FieldCondition", "MatchValue", "Range",
        ):
            getattr(qdrant.models, name).side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        with patch.dict("sys.modules", {"openai": openai, "qdrant_client": qdrant}):
            module = runpy.run_path(str(SCRIPT))
        openai.OpenAI.assert_not_called()
        qdrant.QdrantClient.assert_not_called()
        return module, openai, qdrant

    def test_embed_uses_context_managed_openai(self):
        module, openai, _ = self.load_example()
        api = openai.OpenAI.return_value.__enter__.return_value
        api.embeddings.create.return_value.data = [SimpleNamespace(embedding=[1.0, 2.0])]
        self.assertEqual(module["embed"](["hello"]), [[1.0, 2.0]])
        api.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-small"
        )
        openai.OpenAI.return_value.__exit__.assert_called_once()

    def test_main_creates_only_missing_collection_and_filters(self):
        for exists in (True, False):
            with self.subTest(exists=exists):
                module, _, qdrant = self.load_example()
                client = qdrant.QdrantClient.return_value
                client.collection_exists.return_value = exists
                client.query_points.return_value.points = [
                    SimpleNamespace(score=0.87654, payload={"text": "result"})
                ]
                client.get_collection.return_value.points_count = 5
                embed = MagicMock(side_effect=[[[float(i)] for i in range(5)], [[10.]], [[20.]]])
                output = io.StringIO()
                with patch.dict(module["main"].__globals__, embed=embed):
                    with contextlib.redirect_stdout(output):
                        module["main"]()
                client.recreate_collection.assert_not_called()
                client.delete_collection.assert_not_called()
                self.assertEqual(client.create_collection.call_count, int(not exists))
                if not exists:
                    config = client.create_collection.call_args.kwargs
                    self.assertEqual(config["vectors_config"].size, 1536)
                    self.assertEqual(config["hnsw_config"].m, 16)
                    self.assertEqual(config["hnsw_config"].ef_construct, 200)
                    self.assertEqual(config["optimizers_config"].indexing_threshold, 1000)
                points = client.upsert.call_args.kwargs["points"]
                self.assertEqual([p.id for p in points], list(range(5)))
                self.assertEqual([p.payload for p in points], module["DOCS"])
                self.assertEqual([p.vector for p in points], [[float(i)] for i in range(5)])
                basic, filtered = [call.kwargs for call in client.query_points.call_args_list]
                self.assertEqual(basic["query"], [10.])
                self.assertEqual(filtered["query"], [20.])
                self.assertEqual(basic["limit"], 3)
                self.assertTrue(basic["with_payload"])
                topic, year = filtered["query_filter"].must
                self.assertEqual((topic.key, topic.match.value), ("topic", "vectordb"))
                self.assertEqual((year.key, year.range.gte), ("year", 2024))
                client.create_payload_index.assert_called_once()
                self.assertIn("[0.877] result", output.getvalue())
                self.assertIn("Collection info: 5 points", output.getvalue())
                client.close.assert_called_once()

    def test_connection_closes_on_error(self):
        module, _, qdrant = self.load_example()
        client = qdrant.QdrantClient.return_value
        client.collection_exists.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            module["main"]()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
