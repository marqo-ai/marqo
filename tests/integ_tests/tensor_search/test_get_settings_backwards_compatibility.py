import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from marqo.tensor_search import api
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, UnstructuredMarqoIndex, TextPreProcessing, ImagePreProcessing,
    HnswConfig, Model, DistanceMetric, VectorNumericType, TextSplitMethod
)

class TestGetSettingsBackwardsCompatibility(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_index_management = MagicMock()
        self.mock_config.index_management = self.mock_index_management
        api.app.dependency_overrides[api.get_config] = lambda: self.mock_config
        self.client = TestClient(api.app)

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def create_mock_index(self, index_class, **kwargs):
        common_params = {
            "name": "test_index",
            "schema_name": "test_schema",
            "model": Model(name="test_model", properties=None),
            "normalize_embeddings": True,
            "text_preprocessing": TextPreProcessing(
                split_length=100,
                split_overlap=0,
                split_method=TextSplitMethod.Word
            ),
            "image_preprocessing": ImagePreProcessing(),
            "distance_metric": DistanceMetric.PrenormalizedAngular,
            "vector_numeric_type": VectorNumericType.Float,
            "hnsw_config": HnswConfig(ef_construction=128, m=16),
            "marqo_version": "2.11.0",
            "created_at": 1234567890,
            "updated_at": 1234567890,
            **kwargs
        }
        return index_class(**common_params)

    def assert_common_settings(self, settings):
        expected_fields = [
            "type", "model", "normalizeEmbeddings", "textPreprocessing",
            "imagePreprocessing", "vectorNumericType", "annParameters"
        ]
        for field in expected_fields:
            self.assertIn(field, settings)

        self.assertNotIn("videoPreprocessing", settings)
        self.assertNotIn("audioPreprocessing", settings)

    def test_get_settings_pre_2_12_structured_index(self):
        mock_index = self.create_mock_index(
            StructuredMarqoIndex,
            type="structured",
            fields=[],
            tensor_fields=[]
        )
        self.mock_index_management.get_index.return_value = mock_index

        response = self.client.get("/indexes/test_index/settings")
        self.assertEqual(response.status_code, 200)
        settings = response.json()

        self.assert_common_settings(settings)
        self.assertEqual(settings["type"], "structured")
        self.assertIn("allFields", settings)
        self.assertIn("tensorFields", settings)

    def test_get_settings_pre_2_12_unstructured_index(self):
        mock_index = self.create_mock_index(
            UnstructuredMarqoIndex,
            type="unstructured",
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=200
        )
        self.mock_index_management.get_index.return_value = mock_index

        response = self.client.get("/indexes/test_index/settings")
        self.assertEqual(response.status_code, 200)
        settings = response.json()

        self.assert_common_settings(settings)
        self.assertEqual(settings["type"], "unstructured")
        self.assertIn("treatUrlsAndPointersAsImages", settings)
        self.assertIn("treatUrlsAndPointersAsMedia", settings)
        self.assertIn("filterStringMaxLength", settings)

    def test_get_settings_index_not_found(self):
        from marqo.core.exceptions import IndexNotFoundError
        self.mock_index_management.get_index.side_effect = IndexNotFoundError("Index not found")

        response = self.client.get("/indexes/non_existent_index/settings")
        self.assertEqual(response.status_code, 404)
        self.assertIn("message", response.json())
        self.assertIn("not found", response.json()["message"])