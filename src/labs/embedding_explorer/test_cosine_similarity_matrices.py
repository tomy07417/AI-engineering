"""Offline tests for intra-category and cross-category cosine blocks."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


HERE = Path(__file__).resolve().parent


class CategoryMatrixTests(unittest.TestCase):
    def setUp(self):
        self.generator = types.ModuleType("generate_embeddings")
        self.generator.generate_embeddings = Mock(
            side_effect=AssertionError("Embedding generation must be mocked explicitly")
        )
        spec = importlib.util.spec_from_file_location(
            "lab_similarity", HERE / "cosine_similarity_matrices.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        self.import_output = io.StringIO()
        with patch.dict(sys.modules, {"generate_embeddings": self.generator}), patch.object(
            sys, "path", [str(HERE), *sys.path]
        ), contextlib.redirect_stdout(self.import_output):
            spec.loader.exec_module(self.module)

    def test_import_has_no_external_side_effects_or_output(self):
        self.generator.generate_embeddings.assert_not_called()
        self.assertEqual(self.import_output.getvalue(), "")

    def test_six_blocks_normalization_values_and_first_occurrence_order(self):
        labels = ["sports", "tech", "cooking"] * 4
        vectors = np.array([[1., 0.], [.6, .8], [0., 1.]] * 4)
        for scales in (np.ones((12, 1)), np.arange(1, 13).reshape(-1, 1)):
            with self.subTest(normalized=bool(np.all(scales == 1))):
                embeddings = vectors * scales
                before = embeddings.copy()
                result = self.module.build_category_matrices(embeddings, labels)
                self.assertEqual(set(result), {"intra", "cross"})
                self.assertEqual(list(result["intra"]), ["sports", "tech", "cooking"])
                self.assertEqual(list(result["cross"]), [
                    ("sports", "tech"), ("sports", "cooking"), ("tech", "cooking")
                ])
                for matrix in result["intra"].values():
                    self.assertEqual(matrix.shape, (4, 4))
                    np.testing.assert_allclose(matrix, np.ones((4, 4)))
                    np.testing.assert_allclose(matrix, matrix.T)
                    np.testing.assert_allclose(np.diag(matrix), np.ones(4))
                for pair, value in [
                    (("sports", "tech"), .6),
                    (("sports", "cooking"), 0.),
                    (("tech", "cooking"), .8),
                ]:
                    self.assertEqual(result["cross"][pair].shape, (4, 4))
                    np.testing.assert_allclose(result["cross"][pair], np.full((4, 4), value))
                np.testing.assert_array_equal(embeddings, before)
        self.generator.generate_embeddings.assert_not_called()

    def test_noncontiguous_members_keep_original_row_and_column_order(self):
        labels = ["tech", "cooking", "tech", "sports", "cooking", "sports"]
        embeddings = np.array([
            [1., 0.], [0., 1.], [-1., 0.], [.6, .8], [3., 4.], [0., -2.]
        ])
        result = self.module.build_category_matrices(embeddings, labels)
        expected_intra = {
            "tech": [[1., -1.], [-1., 1.]],
            "cooking": [[1., .8], [.8, 1.]],
            "sports": [[1., -.8], [-.8, 1.]],
        }
        expected_cross = {
            ("tech", "cooking"): [[0., .6], [0., -.6]],
            ("tech", "sports"): [[.6, 0.], [-.6, 0.]],
            ("cooking", "sports"): [[.8, -1.], [1., -.8]],
        }
        for section, expected in (("intra", expected_intra), ("cross", expected_cross)):
            for key, matrix in expected.items():
                np.testing.assert_allclose(result[section][key], matrix, atol=1e-12)

    def test_main_prints_hand_calculated_means_including_diagonal(self):
        labels = ["tech", "cooking", "tech", "sports", "cooking", "sports"]
        embeddings = np.array([
            [1., 0.], [0., 1.], [-1., 0.], [.6, .8], [3., 4.], [0., -2.]
        ])
        self.generator.generate_embeddings.side_effect = None
        self.generator.generate_embeddings.return_value = (embeddings, embeddings)
        output = io.StringIO()
        with patch.object(self.module, "LABELS", labels), contextlib.redirect_stdout(output):
            self.module.main()
        means = [line for line in output.getvalue().splitlines() if line.startswith("Mean:")]
        # E.g. cooking: (1 + .8 + .8 + 1) / 4 = .9, not .8.
        self.assertEqual(means, [
            "Mean: 0.0", "Mean: 0.9", "Mean: 0.1",
            "Mean: 0.0", "Mean: 0.0", "Mean: 0.0",
        ] * 2)

    def test_mean_rounds_to_four_decimal_places(self):
        value = .123456
        embeddings = np.array([[1., 0.], [value, np.sqrt(1 - value ** 2)]])
        self.generator.generate_embeddings.side_effect = None
        self.generator.generate_embeddings.return_value = (embeddings, embeddings)
        output = io.StringIO()
        with patch.object(self.module, "LABELS", ["tech", "cooking"]), contextlib.redirect_stdout(output):
            self.module.main()
        self.assertEqual(output.getvalue().count("Mean: 0.1235\n"), 2)

    def test_main_generates_once_and_prints_independent_blocks_with_original_keys(self):
        # Incompatible feature counts forbid cross-model dot products. The models
        # also have distinct expected scores: all ones vs identity/zero blocks.
        remote = np.tile([3., 4.], (12, 1))
        local = np.eye(12, 13)
        for labels in (self.module.LABELS, ["sports", "tech", "cooking"] * 4):
            with self.subTest(labels=labels):
                self.generator.generate_embeddings.reset_mock()
                self.generator.generate_embeddings.side_effect = None
                self.generator.generate_embeddings.return_value = (remote, local)
                output = io.StringIO()
                options_before = np.get_printoptions()
                with patch.object(self.module, "LABELS", labels), contextlib.redirect_stdout(output), patch.object(
                    self.module, "cosine_similarity", wraps=cosine_similarity, create=True
                ) as similarity:
                    self.module.main()
                self.generator.generate_embeddings.assert_called_once_with()
                self.assertEqual(similarity.call_count, 2)
                for invocation, embeddings in zip(similarity.call_args_list, (remote, local)):
                    self.assertEqual(len(invocation.args), 1)
                    self.assertIs(invocation.args[0], embeddings)
                    self.assertEqual(invocation.kwargs, {})
                categories = list(dict.fromkeys(labels))
                keys = {
                    category: " ".join(
                        f"{index}:{label}" for index, label in enumerate(labels)
                        if label == category
                    ) for category in categories
                }
                expected = []
                with np.printoptions(precision=3, suppress=True):
                    for model in ("OpenAI", "SentenceTransformer"):
                        expected.append(f"{model} cosine similarity")
                        for category in categories:
                            matrix = np.ones((4, 4)) if model == "OpenAI" else np.eye(4)
                            expected.extend([
                                f"Intra-category: {category}",
                                f"Rows: {keys[category]}", f"Columns: {keys[category]}", str(matrix),
                                f"Mean: {1.0 if model == 'OpenAI' else 0.25}",
                            ])
                        for index, left in enumerate(categories):
                            for right in categories[index + 1:]:
                                matrix = np.ones((4, 4)) if model == "OpenAI" else np.zeros((4, 4))
                                expected.extend([
                                    f"Cross-category: {left} vs {right}",
                                    f"Rows: {keys[left]}", f"Columns: {keys[right]}", str(matrix),
                                    f"Mean: {1.0 if model == 'OpenAI' else 0.0}",
                                ])
                self.assertEqual(output.getvalue(), "\n".join(expected) + "\n")
                self.assertEqual(np.get_printoptions(), options_before)


if __name__ == "__main__":
    unittest.main()
