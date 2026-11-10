"""Offline checks: python -m unittest examples.test_qdrant_collection_design -v."""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).with_name("qdrant_collection_design.py")


class CollectionDesignTests(unittest.TestCase):
    def load_example(self):
        sdk = MagicMock()
        for name in ("VectorParams", "HnswConfigDiff", "ScalarQuantization",
                     "ScalarQuantizationConfig", "PointStruct", "FieldCondition",
                     "MatchValue", "Filter"):
            getattr(sdk.models, name).side_effect = lambda **kw: SimpleNamespace(**kw)
        sdk.models.Distance.side_effect = lambda value: value
        with patch.dict("sys.modules", {"qdrant_client": sdk}):
            module = runpy.run_path(str(SCRIPT))
        sdk.QdrantClient.assert_not_called()
        return module, sdk

    def test_schema_configuration_and_existing_collection(self):
        for exists in (True, False):
            with self.subTest(exists=exists):
                module, sdk = self.load_example()
                client = MagicMock()
                client.collection_exists.return_value = exists
                schema = module["SCHEMAS"][0]
                module["create_collection"](client, schema)
                client.recreate_collection.assert_not_called()
                client.delete_collection.assert_not_called()
                self.assertEqual(client.create_collection.call_count, int(not exists))
                if not exists:
                    config = client.create_collection.call_args.kwargs
                    self.assertEqual(config["vectors_config"].size, 1536)
                    self.assertTrue(config["vectors_config"].on_disk)
                    self.assertEqual(config["vectors_config"].distance, "Cosine")
                    self.assertFalse(config["hnsw_config"].on_disk)
                    self.assertEqual(config["hnsw_config"].m, 16)
                    self.assertEqual(config["hnsw_config"].ef_construct, 200)
                    scalar = config["quantization_config"].scalar
                    self.assertEqual(scalar.type, sdk.models.ScalarType.INT8)
                    self.assertTrue(scalar.always_ram)
                indices = {call.kwargs["field_name"]: call.kwargs["field_schema"]
                           for call in client.create_payload_index.call_args_list}
                self.assertEqual(indices["created_at"], sdk.models.PayloadSchemaType.DATETIME)
                self.assertEqual(indices["tenant_id"], sdk.models.PayloadSchemaType.KEYWORD)
                self.assertEqual(len(indices), 4)

    def test_reserved_metadata_rejected_before_write(self):
        module, _ = self.load_example()
        client = MagicMock()
        for metadata in ({"tenant_id": "other"}, {"content": "override"}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    module["upsert_document"](
                        client, "kb", "uuid", [1.], "tenant-a", "hello", metadata
                    )
        client.upsert.assert_not_called()

    def test_upsert_payload_and_tenant_search(self):
        module, _ = self.load_example()
        client = MagicMock()
        metadata = {"language": "en"}
        module["upsert_document"](client, "kb", "uuid", [1.], "tenant-a", "hello", metadata)
        point = client.upsert.call_args.kwargs["points"][0]
        self.assertEqual(point.payload, {
            "tenant_id": "tenant-a", "content": "hello", "language": "en"
        })
        self.assertEqual(point.id, "uuid")
        self.assertEqual(point.vector, [1.])
        self.assertEqual(metadata, {"language": "en"})
        client.query_points.return_value.points = [SimpleNamespace(payload=point.payload, score=.8)]
        results = module["search_tenant"](
            client, "kb", [1.], "tenant-a", top_k=3, filters={"language": "en"}
        )
        call = client.query_points.call_args.kwargs
        self.assertEqual(call["limit"], 3)
        self.assertTrue(call["with_payload"])
        self.assertEqual([(c.key, c.match.value) for c in call["query_filter"].must],
                         [("tenant_id", "tenant-a"), ("language", "en")])
        self.assertEqual(results, [{"content": "hello", "metadata": {"language": "en"}, "score": .8}])
        module["search_tenant"](client, "kb", [1.], "tenant-a")
        conditions = client.query_points.call_args.kwargs["query_filter"].must
        self.assertEqual([(c.key, c.match.value) for c in conditions], [("tenant_id", "tenant-a")])

    def test_main_closes_client_on_error(self):
        module, sdk = self.load_example()
        client = sdk.QdrantClient.return_value
        client.collection_exists.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            module["main"]()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
