"""Offline checks: python -m unittest examples.test_pgvector_search -v."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


class PgvectorSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.asyncpg = ModuleType("asyncpg")
        self.asyncpg.connect = AsyncMock()
        self.openai = ModuleType("openai")
        self.openai.OpenAI = Mock()
        with patch.dict(sys.modules, asyncpg=self.asyncpg, openai=self.openai):
            spec = importlib.util.spec_from_file_location(
                "pgvector_example", Path(__file__).with_name("pgvector_search.py")
            )
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.conn = SimpleNamespace(execute=AsyncMock(), fetch=AsyncMock(), close=AsyncMock())

    def test_import_is_inert(self):
        self.openai.OpenAI.assert_not_called()
        self.asyncpg.connect.assert_not_called()

    def test_embed_uses_small_model(self):
        client = self.openai.OpenAI.return_value
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
        )
        self.assertEqual(self.module.embed(["first", "second"]), [[0.1, 0.2], [0.3, 0.4]])
        client.embeddings.create.assert_called_once_with(
            input=["first", "second"], model="text-embedding-3-small"
        )

    async def test_setup_schema_and_hnsw(self):
        await self.module.setup_database(self.conn)
        statements = [" ".join(call.args[0].split()) for call in self.conn.execute.call_args_list]
        self.assertEqual(statements, [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE TABLE IF NOT EXISTS documents ( id SERIAL PRIMARY KEY, content TEXT, metadata JSONB, embedding vector(1536), created_at TIMESTAMP DEFAULT NOW() )",
            "CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
        ])

    async def test_insert_positional_values(self):
        docs = [{"content": "RAG", "metadata": {"topic": "rag"}},
                {"content": "Database", "metadata": {"topic": "database"}}]
        with patch.object(self.module, "embed", return_value=[[0.1, 0.2], [0.3, 0.4]]) as embed:
            await self.module.insert_documents(self.conn, docs)
        embed.assert_called_once_with(["RAG", "Database"])
        self.assertEqual(self.conn.execute.await_count, 2)
        for call, doc, vector in zip(self.conn.execute.call_args_list, docs, [[0.1, 0.2], [0.3, 0.4]]):
            self.assertEqual(" ".join(call.args[0].split()),
                             "INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2::jsonb, $3::vector)")
            self.assertEqual(call.args[1:], (doc["content"], json.dumps(doc["metadata"]), str(vector)))

    async def test_search_without_filter_defaults_and_decodes_results(self):
        self.conn.fetch.return_value = [{"content": "RAG", "metadata": '{"topic": "rag"}', "similarity": 0.87654321}]
        with patch.object(self.module, "embed", return_value=[[0.1, 0.2]]) as embed:
            result = await self.module.semantic_search(self.conn, "vector search technology")
        embed.assert_called_once_with(["vector search technology"])
        sql, *params = self.conn.fetch.call_args.args
        self.assertEqual(" ".join(sql.split()),
                         "SELECT content, metadata, 1 - (embedding <=> $1::vector) AS similarity FROM documents ORDER BY embedding <=> $1::vector LIMIT $2")
        self.assertEqual(params, ["[0.1, 0.2]", 5])
        self.assertEqual(result, [{"content": "RAG", "metadata": {"topic": "rag"}, "similarity": 0.8765}])

    async def test_search_parameterizes_keys_values_and_limit(self):
        malicious_key = "topic' OR TRUE; DROP TABLE documents; --"
        self.conn.fetch.return_value = []
        with patch.object(self.module, "embed", return_value=[[0.5]]):
            result = await self.module.semantic_search(
                self.conn, "algorithms", top_k=2,
                metadata_filter={malicious_key: "algorithms", "level": "1"},
            )
        sql, *params = self.conn.fetch.call_args.args
        self.assertNotIn(malicious_key, sql)
        self.assertNotIn("algorithms", sql)
        self.assertIn("WHERE metadata->> $2::text = $3::text AND metadata->> $4::text = $5::text", sql)
        self.assertTrue(sql.rstrip().endswith("ORDER BY embedding <=> $1::vector LIMIT $6"))
        self.assertEqual(params, ["[0.5]", malicious_key, "algorithms", "level", "1", 2])
        self.assertEqual(result, [])

    async def test_empty_filter_preserves_unfiltered_search(self):
        self.conn.fetch.return_value = []
        with patch.object(self.module, "embed", return_value=[[0.5]]):
            await self.module.semantic_search(self.conn, "query", top_k=2, metadata_filter={})
        sql, *params = self.conn.fetch.call_args.args
        self.assertNotIn("WHERE", sql)
        self.assertTrue(sql.rstrip().endswith("LIMIT $2"))
        self.assertEqual(params, ["[0.5]", 2])

    async def test_main_demonstrates_searches_and_closes(self):
        self.asyncpg.connect.return_value = self.conn
        with patch.object(self.module, "setup_database", new_callable=AsyncMock), \
             patch.object(self.module, "insert_documents", new_callable=AsyncMock) as insert, \
             patch.object(self.module, "semantic_search", new_callable=AsyncMock, return_value=[]) as search:
            await self.module.main()
        self.asyncpg.connect.assert_awaited_once_with("postgresql://localhost/ai_service")
        docs = insert.call_args.args[1]
        self.assertEqual([doc["metadata"]["topic"] for doc in docs], ["rag", "database", "algorithms"])
        self.assertEqual(search.call_args_list[0].args, (self.conn, "vector search technology"))
        self.assertEqual(search.call_args_list[0].kwargs, {"top_k": 2})
        self.assertEqual(search.call_args_list[1].args, (self.conn, "algorithms"))
        self.assertEqual(search.call_args_list[1].kwargs, {"top_k": 2, "metadata_filter": {"topic": "algorithms"}})
        self.conn.close.assert_awaited_once_with()

    async def test_main_closes_connection_on_failure(self):
        self.asyncpg.connect.return_value = self.conn
        with patch.object(self.module, "setup_database", new_callable=AsyncMock, side_effect=RuntimeError("setup failed")):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                await self.module.main()
        self.conn.close.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
