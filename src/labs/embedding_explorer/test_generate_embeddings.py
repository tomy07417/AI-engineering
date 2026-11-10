"""Offline checks: python -m unittest discover -s src/labs/embedding_explorer -p 'test_generate_embeddings.py' -v"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, call, patch

import numpy as np


HERE = Path(__file__).resolve().parent


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.openai = types.ModuleType("openai")
        self.openai.OpenAI = Mock()
        self.local = types.ModuleType("sentence_transformers")
        self.local.SentenceTransformer = Mock()
        self.dotenv = types.ModuleType("dotenv")
        self.dotenv.load_dotenv = Mock()
        modules = {
            "openai": self.openai,
            "sentence_transformers": self.local,
            "dotenv": self.dotenv,
        }
        spec = importlib.util.spec_from_file_location(
            "lab_embeddings", HERE / "generate_embeddings.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(sys, "path", [str(HERE), *sys.path]):
            spec.loader.exec_module(self.module)

    def test_import_has_no_external_side_effects(self):
        self.dotenv.load_dotenv.assert_not_called()
        self.openai.OpenAI.assert_not_called()
        self.local.SentenceTransformer.assert_not_called()

    def test_missing_or_blank_key_stops_before_clients(self):
        for environment in ({}, {"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": " \t "}):
            with self.subTest(environment=environment), patch.dict(
                self.module.os.environ, environment, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "^Define OPENAI_API_KEY en el archivo .env o en el entorno antes de generar embeddings\\.$"):
                    self.module.generate_embeddings()
                self.openai.OpenAI.assert_not_called()
                self.local.SentenceTransformer.assert_not_called()

    def prepare_success(self):
        events = Mock()
        events.attach_mock(self.dotenv.load_dotenv, "dotenv")
        events.attach_mock(self.openai.OpenAI, "openai")
        events.attach_mock(self.local.SentenceTransformer, "local")
        self.openai.OpenAI.return_value.embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for _ in range(12)]
        )
        self.local.SentenceTransformer.return_value.encode.return_value = np.tile(
            [0.6, 0.8], (12, 1)
        )
        return events

    def test_models_order_normalization_and_arrays(self):
        events = self.prepare_success()
        with patch.dict(self.module.os.environ, {"OPENAI_API_KEY": "test-placeholder"}, clear=True):
            remote, local = self.module.generate_embeddings()
        self.assertEqual(events.mock_calls, [
            call.dotenv(HERE / ".env", override=False),
            call.openai(api_key="test-placeholder"),
            call.openai().embeddings.create(model="text-embedding-3-small", input=self.module.TEXTS),
            call.local("all-MiniLM-L6-v2"),
            call.local().encode(self.module.TEXTS, normalize_embeddings=True),
        ])
        self.assertIsInstance(remote, np.ndarray)
        self.assertIsInstance(local, np.ndarray)
        self.assertEqual(remote.shape, (12, 3))
        self.assertEqual(local.shape, (12, 2))
        np.testing.assert_allclose(remote, np.tile([1.0, 0.0, 0.0], (12, 1)))
        np.testing.assert_allclose(np.linalg.norm(local, axis=1), np.ones(12))

    def test_main_prints_only_labeled_shapes(self):
        self.prepare_success()
        with patch.dict(self.module.os.environ, {"OPENAI_API_KEY": "test-placeholder"}, clear=True), patch("builtins.print") as output:
            self.module.main()
        self.assertEqual(output.call_args_list, [
            call("OpenAI: (12, 3)"),
            call("SentenceTransformer: (12, 2)"),
        ])


if __name__ == "__main__":
    unittest.main()
