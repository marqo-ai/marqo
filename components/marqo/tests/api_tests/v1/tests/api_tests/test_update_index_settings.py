import json
import uuid

import requests

from tests.marqo_test import MarqoTestCase


class TestUpdateIndexSettingsDryRun(MarqoTestCase):
    """
    Test all the dry-run functionalities of updating index settings. Real updating are not performed in these tests
    as the order of these tests are not guaranteed, and performing real updates may affect other tests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.original_model_properties = {
            "isMarqtuneModel": True,
            "url": "https://invalida-model-properties/model",
            "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
            "dimensions": 512,
            "type": "open_clip"
        }

        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "type": "unstructured",
                "model": "test-model",
                "modelProperties": {
                    "isMarqtuneModel": True,
                    "url": "https://invalida-model-properties/model",
                    "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                    "dimensions": 512,
                    "type": "open_clip"
                }
            }
        ])

        cls.model_properties = {
            "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dimensions": 512,
            "type": "open_clip",
            "tritonImageEncoderProperties": {
                "maxBatchSize": 8,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-image-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/image-encoder/model.onnx"
                ],
                "input": [
                    {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
            "tritonTextEncoderProperties": {
                "maxBatchSize": 16,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-text-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/text-encoder/model.onnx",
                ],
                "input": [{"name": "input", "dims": [77], "dataType": "TYPE_INT32"}],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
        }

        cls.indexes_to_delete = [cls.index_name]

    def test_dry_run_invalid_index_updating(self):
        test_cases = [
            (
                "Invalid: Not keeping index_settings backwards compatible", self.model_properties
            ),
            (
                "Invalid: Changing model dimensions", {**self.model_properties, "dimensions": 1024},
            ),
            (
                "Invalid: Changing model type", {**self.model_properties, "type": "different_type"}
            )
        ]

        for description, new_model_properties in test_cases:
            for force in ["true", "false"]:
                with self.subTest(description=description, force=force):
                    response = requests.patch(
                        f"{self._MARQO_URL}/indexes/{self.index_name}/index-settings?dry_run=true&force={force}",
                        json={
                            "modelProperties": new_model_properties,
                        }
                    ).json()

                    self.assertEqual(True, response["error"])
                    self.assertEqual(False, response["updated"])

    def test_dry_run_valid_index_updating(self):
        target_model_properties = {**self.original_model_properties, **self.model_properties}
        res = requests.patch(
            f"{self._MARQO_URL}/indexes/{self.index_name}/index-settings?dry_run=true",
            json={
                "modelProperties": target_model_properties,
            }
        ).json()

        self.assertEqual(False, res["error"])
        self.assertEqual(False, res["updated"])
        self.assertEqual(self.original_model_properties, res["oldSettings"]["modelProperties"])
        self.assertEqual(target_model_properties, res["newSettings"]["modelProperties"])
        self.assertEqual("Dry run - no changes deployed", res["reason"])


class TestUpdateIndexSettingsRealRun(MarqoTestCase):
    """
    Test that updating index settings works as expected in real runs. dry_run=false and force=false
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.original_model_properties = {
            "isMarqtuneModel": True,
            "url": "https://invalida-model-properties/model",
            "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
            "dimensions": 512,
            "type": "open_clip"
        }

        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "type": "unstructured",
                "model": "test-model",
                "modelProperties": {
                    "isMarqtuneModel": True,
                    "url": "https://invalida-model-properties/model",
                    "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                    "dimensions": 512,
                    "type": "open_clip"
                }
            }
        ])

        cls.model_properties = {
            "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dimensions": 512,
            "type": "open_clip",
            "tritonImageEncoderProperties": {
                "maxBatchSize": 8,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-image-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/image-encoder/model.onnx"
                ],
                "input": [
                    {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
            "tritonTextEncoderProperties": {
                "maxBatchSize": 16,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-text-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/text-encoder/model.onnx",
                ],
                "input": [{"name": "input", "dims": [77], "dataType": "TYPE_INT32"}],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
        }

        cls.indexes_to_delete = [cls.index_name]

    def test_dry_run_valid_index_updating(self):
        target_model_properties = {**self.original_model_properties, **self.model_properties}
        res = requests.patch(
            f"{self._MARQO_URL}/indexes/{self.index_name}/index-settings?dry_run=false",
            json={
                "modelProperties": target_model_properties,
            }
        ).json()

        self.assertEqual(False, res["error"])
        self.assertEqual(True, res["updated"])
        self.assertEqual(self.original_model_properties, res["oldSettings"]["modelProperties"])
        self.assertEqual(target_model_properties, res["newSettings"]["modelProperties"])
        self.assertEqual("Settings updated successfully", res["reason"])

        _ = self.client.index(self.index_name).search(
            "Test query after dry run", search_method="TENSOR"
        )

        loaded_models = requests.get(f"{self._MARQO_URL}/models?detailed=true").json()
        self.assertEqual("test-model||14e4", loaded_models["models"][0]["modelName"])
        # We can't check more details as isMarqtuneModel is True, so the modelProperties are not shown in detail


class TestUpdateIndexSettingsRealRunForceTrue(MarqoTestCase):
    """
    Test that updating index settings works as expected in real runs. dry_run=false and force=True
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.original_model_properties = {
            "isMarqtuneModel": True,
            "url": "https://invalida-model-properties/model",
            "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
            "dimensions": 512,
            "type": "open_clip"
        }

        cls.create_indexes([
            {
                "indexName": cls.index_name,
                "type": "unstructured",
                "model": "test-model",
                "modelProperties": {
                    "isMarqtuneModel": True,
                    "url": "https://invalida-model-properties/model",
                    "name": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                    "dimensions": 512,
                    "type": "open_clip"
                }
            }
        ])

        cls.model_properties = {
            "name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dimensions": 512,
            "type": "open_clip",
            "tritonImageEncoderProperties": {
                "maxBatchSize": 8,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-image-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/image-encoder/model.onnx"
                ],
                "input": [
                    {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
            "tritonTextEncoderProperties": {
                "maxBatchSize": 16,
                "name": "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-text-encoder",
                "sources": [
                    "s3://marqo-opensource-models/laion-CLIP-ViT-B-32-laion2B-s34B-b79K/text-encoder/model.onnx",
                ],
                "input": [{"name": "input", "dims": [77], "dataType": "TYPE_INT32"}],
                "output": [{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
            },
        }

        cls.indexes_to_delete = [cls.index_name]

    def test_dry_run_valid_index_updating(self):
        target_model_properties = self.model_properties
        res = requests.patch(
            f"{self._MARQO_URL}/indexes/{self.index_name}/index-settings?dry_run=false&force=true",
            json={
                "modelProperties": target_model_properties,
            }
        ).json()

        self.assertEqual(True, res["error"])
        self.assertEqual(True, res["updated"])
        self.assertEqual(self.original_model_properties, res["oldSettings"]["modelProperties"])
        self.assertEqual(target_model_properties, res["newSettings"]["modelProperties"])
        self.assertIn(
            "Update forced despite validation errors:",
            res["reason"]
        )

        _ = self.client.index(self.index_name).search(
            "Test query after dry run", search_method="TENSOR"
        )

        loaded_models = requests.get(f"{self._MARQO_URL}/models?detailed=true").json()
        self.assertEqual("test-model||632c", loaded_models["models"][0]["modelName"])
        loaded_model_properties = json.loads(loaded_models["models"][0]["modelProperties"])
        # There is some None and added default values in loaded_model_properties, so we check only the keys we set
        for k, v in target_model_properties.items():
            self.assertEqual(
                v, loaded_model_properties[k], f"Mismatch in model property key: {k}. "
            )