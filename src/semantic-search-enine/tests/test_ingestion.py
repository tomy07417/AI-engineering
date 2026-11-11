"""Offline tests for the four public ingestion functions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from semantic_search_engine import generate_embedding, ingest, load_document, upsert_document


class LoaderTests(unittest.TestCase):
    def test_reads_txt_and_markdown_with_relative_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manuals").mkdir()
            for suffix in ("txt", "md"):
                path = root / "manuals" / f"guide.{suffix}"
                path.write_text("# Guide\nHola", encoding="utf-8")
                document = load_document(path, root=root)
                self.assertEqual(document, {
                    "title": f"manuals/guide.{suffix}", "content": "# Guide\nHola",
                    "metadata": {"source": f"manuals/guide.{suffix}"},
                })
                self.assertEqual(load_document(path)["title"], f"guide.{suffix}")

    def test_rejects_unsupported_blank_missing_and_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (("file.pdf", "text"), ("blank.txt", " \n")):
                path = root / name
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_document(path)
            with self.assertRaises(ValueError):
                load_document(root / "missing.txt")
            path = root / "file.txt"
            path.write_text("hello", encoding="utf-8")
            (root / "subdir").mkdir()
            with self.assertRaises(ValueError):
                load_document(path, root=root / "subdir")


class EmbeddingTests(unittest.TestCase):
    def test_loads_local_model_and_normalizes_batch(self):
        sdk = MagicMock()
        sdk.SentenceTransformer.return_value.encode.return_value = np.ones((2, 384))
        with patch.dict("sys.modules", {"sentence_transformers": sdk}):
            vectors = generate_embedding(["Alpha", "Beta"], cache_folder="model-cache")
        sdk.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2", cache_folder="model-cache")
        sdk.SentenceTransformer.return_value.encode.assert_called_once_with(
            ["Alpha", "Beta"], batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self.assertEqual(vectors.shape, (2, 384))
        self.assertEqual(vectors.dtype, np.float32)

    def test_reuses_injected_model_and_rejects_invalid_vectors(self):
        model = MagicMock()
        for values in (np.ones((1, 12)), np.full((1, 384), np.nan)):
            with self.subTest(shape=values.shape):
                model.encode.return_value = values
                with self.assertRaisesRegex(ValueError, "384|finite"):
                    generate_embedding(["Alpha"], model=model)


class IngestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = MagicMock()
        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock()
        transaction.__aexit__ = AsyncMock(return_value=False)
        self.transaction = transaction
        self.connection.transaction.return_value = transaction
        self.connection.fetch = AsyncMock(return_value=[])
        self.connection.execute = AsyncMock()
        self.embedding = patch("semantic_search_engine.ingestion.generate_embedding")
        self.generate = self.embedding.start()
        self.generate.side_effect = lambda texts, **kwargs: np.ones((len(texts), 384), dtype=np.float32)
        self.addCleanup(self.embedding.stop)

    async def test_inserts_new_documents_via_parameterized_upsert(self):
        result = await ingest([
            {"title": "a", "content": "Alpha", "metadata": {"topic": "tech"}},
            {"title": "b", "content": "Beta"},
        ], self.connection)
        self.assertEqual(result, {"inserted": 2, "updated": 0, "unchanged": 0, "embedded": 2})
        self.assertEqual(self.generate.call_args.args, (["Alpha", "Beta"],))
        self.assertEqual(self.connection.execute.await_count, 2)
        sql, title, content, vector, metadata = self.connection.execute.await_args_list[0].args
        self.assertIn("ON CONFLICT (title) DO UPDATE", sql)
        self.assertIn("updated_at = NOW()", sql)
        self.assertEqual((title, content, json.loads(metadata)), ("a", "Alpha", {"topic": "tech"}))
        self.assertEqual(len(json.loads(vector)), 384)

    async def test_incremental_only_embeds_new_or_changed_content(self):
        old_vector = json.dumps([0.5] * 384)
        self.connection.fetch.return_value = [
            {"title": "same", "content": "Same", "metadata": "{}", "embedding": old_vector},
            {"title": "changed", "content": "Old", "metadata": "{}", "embedding": old_vector},
            {"title": "metadata", "content": "Keep", "metadata": '{"topic":"old"}', "embedding": old_vector},
        ]
        result = await ingest([
            {"title": "same", "content": "Same"},
            {"title": "changed", "content": "New"},
            {"title": "metadata", "content": "Keep", "metadata": {"topic": "new"}},
            {"title": "new", "content": "Fresh"},
        ], self.connection)
        self.assertEqual(result, {"inserted": 1, "updated": 2, "unchanged": 1, "embedded": 2})
        self.assertEqual(self.generate.call_args.args, (["New", "Fresh"],))
        calls = self.connection.execute.await_args_list
        self.assertEqual(len(calls), 3)
        metadata_call = next(call for call in calls if call.args[1] == "metadata")
        self.assertEqual(json.loads(metadata_call.args[3]), [0.5] * 384)
        self.assertEqual(json.loads(metadata_call.args[4]), {"topic": "new"})

    async def test_unchanged_does_not_load_model_or_write(self):
        self.connection.fetch.return_value = [{
            "title": "a", "content": "Alpha", "metadata": "{}",
            "embedding": json.dumps([1.] * 384),
        }]
        result = await ingest([{"title": "a", "content": "Alpha"}], self.connection)
        self.assertEqual(result, {"inserted": 0, "updated": 0, "unchanged": 1, "embedded": 0})
        self.generate.assert_not_called()
        self.connection.execute.assert_not_called()

    async def test_missing_embedding_is_repaired(self):
        self.connection.fetch.return_value = [{
            "title": "a", "content": "Alpha", "metadata": "{}", "embedding": None,
        }]
        result = await ingest([{"title": "a", "content": "Alpha"}], self.connection)
        self.assertEqual((result["updated"], result["embedded"]), (1, 1))

    async def test_rejects_invalid_or_duplicate_documents_before_database_work(self):
        for documents in (
            [{"title": "", "content": "text"}], [{"title": "a", "content": " "}],
            [{"title": "a", "content": "text", "metadata": []}],
            [{"title": "a", "content": "text", "metadata": {"n": float("nan")}}],
            [{"title": "a", "content": "one"}, {"title": "a", "content": "two"}],
        ):
            with self.subTest(documents=documents):
                with self.assertRaises(ValueError):
                    await ingest(documents, self.connection)
        self.connection.transaction.assert_not_called()
        self.connection.execute.assert_not_called()
        self.generate.assert_not_called()

    async def test_write_failure_propagates_through_transaction(self):
        self.connection.execute.side_effect = RuntimeError("database failure")
        with self.assertRaisesRegex(RuntimeError, "database failure"):
            await ingest([{"title": "a", "content": "Alpha"}], self.connection)
        self.assertIs(self.transaction.__aexit__.await_args.args[0], RuntimeError)

    async def test_upsert_validates_before_write_and_empty_ingest_is_noop(self):
        with self.assertRaises(ValueError):
            await upsert_document(self.connection, {"title": "a", "content": "Alpha"}, [1.])
        result = await ingest([], self.connection)
        self.assertEqual(sum(result.values()), 0)
        self.connection.execute.assert_not_called()
        self.connection.transaction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
