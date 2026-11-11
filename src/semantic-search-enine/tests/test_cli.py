"""Offline public CLI tests."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from semantic_search_engine.__main__ import main


class CliTests(unittest.TestCase):
    def test_help_needs_no_database_or_model(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn(".txt", output.getvalue())
        self.assertIn("--model-cache", output.getvalue())

    def test_missing_database_url_is_rejected_before_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "file.txt").write_text("hello", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as error:
                with patch("semantic_search_engine.__main__.asyncpg.connect", new_callable=AsyncMock) as connect:
                    with self.assertRaises(SystemExit) as result:
                        main([directory])
            self.assertEqual(result.exception.code, 2)
            self.assertIn("DATABASE_URL", error.getvalue())
            connect.assert_not_called()

    def test_recursive_load_success_output_and_connection_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sub").mkdir()
            (root / "a.txt").write_text("Alpha", encoding="utf-8")
            (root / "sub" / "b.md").write_text("Beta", encoding="utf-8")
            (root / "skip.pdf").write_text("ignored", encoding="utf-8")
            connection = AsyncMock()
            stats = {"inserted": 2, "updated": 0, "unchanged": 0, "embedded": 2}
            with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test-only"}):
                with patch("semantic_search_engine.__main__.asyncpg.connect", new_callable=AsyncMock, return_value=connection):
                    with patch("semantic_search_engine.__main__.ingest", new_callable=AsyncMock, return_value=stats) as ingest:
                        with contextlib.redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(main([directory]), 0)
            documents = ingest.await_args.args[0]
            self.assertEqual([doc["title"] for doc in documents], ["a.txt", "sub/b.md"])
            self.assertIs(ingest.await_args.args[1], connection)
            self.assertEqual(output.getvalue(), json.dumps(stats, sort_keys=True) + "\n")
            connection.close.assert_awaited_once()

    def test_compose_connection_defaults_and_overrides(self):
        cases = [
            ({"POSTGRES_PASSWORD": "test-only"},
             "postgresql://semantic_search:test-only@127.0.0.1:5432/semantic_search"),
            ({"POSTGRES_PASSWORD": "p@ss:/", "POSTGRES_USER": "custom-user",
              "POSTGRES_DB": "custom-db", "POSTGRES_PORT": "5544"},
             "postgresql://custom-user:p%40ss%3A%2F@127.0.0.1:5544/custom-db"),
            ({"POSTGRES_PASSWORD": "test-only", "DATABASE_URL": "postgresql://explicit"},
             "postgresql://explicit"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "file.txt").write_text("hello", encoding="utf-8")
            for environment, expected_url in cases:
                with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                    with patch("semantic_search_engine.__main__.asyncpg.connect", new_callable=AsyncMock, return_value=AsyncMock()) as connect:
                        with patch("semantic_search_engine.__main__.ingest", new_callable=AsyncMock, return_value={"inserted": 1}):
                            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as errors:
                                self.assertEqual(main([directory]), 0)
                        connect.assert_awaited_once_with(expected_url)
                    self.assertEqual(output.getvalue(), '{"inserted": 1}\n')
                    self.assertEqual(errors.getvalue(), "")

    def test_empty_directory_and_blank_document_are_rejected_without_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("semantic_search_engine.__main__.asyncpg.connect", new_callable=AsyncMock) as connect:
                with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test-only"}):
                    for blank in (False, True):
                        if blank:
                            (Path(directory) / "blank.txt").write_text(" ", encoding="utf-8")
                        with self.subTest(blank=blank), contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as result:
                                main([directory])
                            self.assertEqual(result.exception.code, 2)
                connect.assert_not_called()

    def test_database_failure_redacts_details_and_closes_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.txt").write_text("Alpha", encoding="utf-8")
            connection = AsyncMock()
            with patch.dict(os.environ, {"DATABASE_URL": "postgresql://secret"}):
                with patch("semantic_search_engine.__main__.asyncpg.connect", new_callable=AsyncMock, return_value=connection):
                    with patch("semantic_search_engine.__main__.ingest", new_callable=AsyncMock, side_effect=RuntimeError("secret")):
                        with contextlib.redirect_stderr(io.StringIO()) as error, contextlib.redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(main([directory]), 1)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(error.getvalue(), "Ingestion failed. Check the database, schema, and model availability.\n")
            self.assertNotIn("secret", error.getvalue())
            connection.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
