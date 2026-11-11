"""Opt-in integration tests. Use only a disposable SEMANTIC_SEARCH_TEST_DSN.

Each test creates/drops its own schema. Enable SEMANTIC_SEARCH_REAL_MODEL=1
for an actual MiniLM download/inference check; default tests never download it.
"""

import json
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import asyncpg
import numpy as np

from semantic_search_engine import ingest

DSN = os.environ.get("SEMANTIC_SEARCH_TEST_DSN")


@unittest.skipUnless(DSN, "Set SEMANTIC_SEARCH_TEST_DSN to a disposable PostgreSQL database")
class PostgresTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = await asyncpg.connect(DSN)
        self.schema = "ingestion_test_" + uuid.uuid4().hex
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        sql = (Path(__file__).resolve().parents[1] / "init.sql").read_text()
        await self.connection.execute(sql)
        self.model = MagicMock()
        self.model.encode.side_effect = lambda texts, **kwargs: np.full(
            (len(texts), 384), 1 / np.sqrt(384), dtype=np.float32,
        )

    async def asyncTearDown(self):
        try:
            await self.connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')
        finally:
            await self.connection.close()

    async def snapshot(self):
        return [dict(row) for row in await self.connection.fetch(
            "SELECT id, title, content, metadata, embedding::text AS embedding, "
            "created_at, updated_at FROM documents ORDER BY title"
        )]

    async def test_incremental_persistence_and_metadata_only_updates(self):
        docs = [{"title": "a.txt", "content": "Alpha"}, {"title": "b.md", "content": "Beta"}]
        result = await ingest(docs, self.connection, model=self.model)
        self.assertEqual(result, {"inserted": 2, "updated": 0, "unchanged": 0, "embedded": 2})
        before = await self.snapshot()
        result = await ingest(docs, self.connection, model=self.model)
        self.assertEqual(result, {"inserted": 0, "updated": 0, "unchanged": 2, "embedded": 0})
        self.assertEqual(await self.snapshot(), before)
        self.assertEqual(self.model.encode.call_count, 1)

        result = await ingest([
            {"title": "a.txt", "content": "Changed"},
            {"title": "b.md", "content": "Beta", "metadata": {"topic": "new"}},
            {"title": "c.txt", "content": "Gamma"},
        ], self.connection, model=self.model)
        self.assertEqual(result, {"inserted": 1, "updated": 2, "unchanged": 0, "embedded": 2})
        after = await self.snapshot()
        self.assertEqual(len(after), 3)
        self.assertEqual(self.model.encode.call_args.args[0], ["Changed", "Gamma"])
        for old, updated in zip(before, after[:2]):
            self.assertEqual((old["id"], old["created_at"]), (updated["id"], updated["created_at"]))
            self.assertGreaterEqual(updated["updated_at"], old["updated_at"])
        self.assertEqual(after[0]["content"], "Changed")
        self.assertEqual(json.loads(after[1]["metadata"]), {"topic": "new"})
        self.assertEqual(after[1]["embedding"], before[1]["embedding"])
        self.assertTrue(all(len(json.loads(row["embedding"])) == 384 for row in after))

    async def test_failed_batch_rolls_back_and_rejected_input_changes_nothing(self):
        await ingest([{"title": "seed", "content": "Seed"}], self.connection, model=self.model)
        before = await self.snapshot()
        await self.connection.execute(
            "ALTER TABLE documents ADD CONSTRAINT reject_test_title CHECK (title <> 'rejected.txt')"
        )
        with self.assertRaises(asyncpg.CheckViolationError):
            await ingest([
                {"title": "accepted.txt", "content": "Valid"},
                {"title": "rejected.txt", "content": "Rejected by test constraint"},
            ], self.connection, model=self.model)
        self.assertEqual(await self.snapshot(), before)
        calls = self.model.encode.call_count
        with self.assertRaises(ValueError):
            await ingest([{"title": "seed", "content": " "}], self.connection, model=self.model)
        self.assertEqual(await self.snapshot(), before)
        self.assertEqual(self.model.encode.call_count, calls)

    @unittest.skipUnless(os.environ.get("SEMANTIC_SEARCH_REAL_MODEL") == "1", "Opt in to real model inference")
    async def test_real_minilm_vectors_are_persisted(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cpu",
            cache_folder=os.environ.get("SEMANTIC_SEARCH_MODEL_CACHE"),
        )
        result = await ingest([
            {"title": "real.txt", "content": "Vector search finds related documents by meaning."},
        ], self.connection, model=model)
        self.assertEqual(result["embedded"], 1)
        rows = await self.snapshot()
        self.assertEqual(len(rows), 1)
        vector = np.asarray(json.loads(rows[0]["embedding"]))
        self.assertEqual(vector.shape, (384,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
