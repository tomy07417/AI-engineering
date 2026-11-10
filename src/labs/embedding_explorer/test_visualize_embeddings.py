"""Offline projection and headless plotting tests."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# Load compiled dependencies before patch.dict restores the module registry.
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        generator = types.ModuleType("generate_embeddings")
        self.generate = generator.generate_embeddings = Mock()
        umap = types.ModuleType("umap")
        umap.UMAP = Mock()
        spec = importlib.util.spec_from_file_location(
            "lab_visualization", HERE / "visualize_embeddings.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        self.output = io.StringIO()
        with patch.dict(sys.modules, {"generate_embeddings": generator, "umap": umap}), patch.object(
            sys, "path", [str(HERE), *sys.path]
        ), patch.object(Path, "mkdir") as mkdir, contextlib.redirect_stdout(self.output):
            spec.loader.exec_module(self.module)
        mkdir.assert_not_called()
        self.addCleanup(plt.close, "all")

    def test_import_is_passive(self):
        self.generate.assert_not_called()
        self.assertEqual(self.output.getvalue(), "")

    def test_reducers_parameters_shapes_and_input_preserved(self):
        for n in (3, 6, 12):
            with self.subTest(n=n), patch.object(self.module, "TSNE") as tsne, patch.object(
                self.module, "UMAP"
            ) as umap:
                embeddings = np.arange(n * 7).reshape(n, 7).astype(float)
                before = embeddings.copy()
                coords = np.arange(n * 2).reshape(n, 2)
                tsne.return_value.fit_transform.return_value = coords
                umap.return_value.fit_transform.return_value = coords + 1
                result = self.module.reduce_embeddings(embeddings)
                tsne.assert_called_once_with(n_components=2, metric="cosine", init="random",
                                             learning_rate="auto", perplexity=min(5, n - 1), random_state=42)
                umap.assert_called_once_with(n_components=2, metric="cosine", init="random",
                                             n_neighbors=min(5, n - 1), random_state=42, n_jobs=1)
                for reducer in (tsne, umap):
                    self.assertIs(reducer.return_value.fit_transform.call_args.args[0], embeddings)
                self.assertEqual(set(result), {"tsne", "umap"})
                np.testing.assert_array_equal(result["tsne"], coords)
                np.testing.assert_array_equal(result["umap"], coords + 1)
                np.testing.assert_array_equal(embeddings, before)

    def test_too_few_documents_rejected_before_fitting(self):
        for n in (0, 1, 2):
            with self.subTest(n=n), patch.object(self.module, "TSNE") as tsne, patch.object(
                self.module, "UMAP"
            ) as umap:
                with self.assertRaisesRegex(ValueError, "At least 3 texts are required"):
                    self.module.reduce_embeddings(np.ones((n, 4)))
                tsne.assert_not_called()
                umap.assert_not_called()

    def test_four_panels_category_colors_legends_and_png(self):
        labels = ["sports", "tech", "cooking"] * 4
        coords = np.arange(24).reshape(12, 2)
        projections = {
            model: {"tsne": coords + offset, "umap": coords + offset + 1}
            for model, offset in (("OpenAI", 0), ("SentenceTransformer", 100))
        }
        with TemporaryDirectory(dir=HERE) as temporary, patch.object(
            self.module.plt, "close", wraps=plt.close
        ) as close:
            path = Path(temporary) / "plots" / "embedding_projections.png"
            self.module.plot_projections(projections, labels, path)
            # Inspect the real figure through the close call (it must be closed after saving).
            figure = close.call_args.args[0]
            self.assertEqual(len(figure.axes), 4)
            colors = {}
            for axis, (model, method) in zip(figure.axes, (
                (model, method) for model in projections for method in ("tsne", "umap")
            )):
                self.assertEqual(axis.get_title(), f"{model} — {'t-SNE' if method == 'tsne' else 'UMAP'}")
                self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                                 ["sports", "tech", "cooking"])
                for category, collection in zip(dict.fromkeys(labels), axis.collections):
                    indices = [i for i, label in enumerate(labels) if label == category]
                    np.testing.assert_array_equal(collection.get_offsets(), projections[model][method][indices])
                    color = collection.get_facecolors()
                    if category in colors:
                        np.testing.assert_array_equal(color, colors[category])
                    colors[category] = color
            self.assertEqual(len({tuple(color[0]) for color in colors.values()}), 3)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertFalse(plt.fignum_exists(figure.number))

    def test_main_generates_once_and_projects_models_independently(self):
        remote, local = np.ones((12, 7)), np.ones((12, 3))
        self.generate.return_value = (remote, local)
        first, second = {"tsne": Mock(), "umap": Mock()}, {"tsne": Mock(), "umap": Mock()}
        with patch.object(self.module, "reduce_embeddings", side_effect=[first, second]) as reduce, patch.object(
            self.module, "plot_projections"
        ) as plot, contextlib.redirect_stdout(self.output):
            self.module.main()
        self.generate.assert_called_once_with()
        self.assertEqual(reduce.call_count, 2)
        self.assertIs(reduce.call_args_list[0].args[0], remote)
        self.assertIs(reduce.call_args_list[1].args[0], local)
        destination = HERE / "plots" / "embedding_projections.png"
        plot.assert_called_once_with({"OpenAI": first, "SentenceTransformer": second}, self.module.LABELS, destination)
        self.assertEqual(self.output.getvalue(), f"{destination}\n")


if __name__ == "__main__":
    unittest.main()
