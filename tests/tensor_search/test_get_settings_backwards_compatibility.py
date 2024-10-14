import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from marqo.tensor_search import api
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, UnstructuredMarqoIndex, TextPreProcessing, ImagePreProcessing,
    HnswConfig, Model, DistanceMetric, VectorNumericType, TextSplitMethod
)
from marqo.tensor_search.api import get_config

class TestGetSettingsBackwardsCompatibility(unittest.TestCase):
    def setUp(self):
        # Set up the test client and mock configuration
        self.client = TestClient(api.app)
        self.mock_config = MagicMock()
        # Override the get_config dependency with our mock
        self.client.app.dependency_overrides[get_config] = lambda: self.mock_config

    def tearDown(self):
        # Clear the dependency overrides after each test
        self.client.app.dependency_overrides.clear()

    def create_mock_index(self, index_class, **kwargs):
        # Helper method to create a mock index with common parameters
        common_params = {
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
            **kwargs  # Allow overriding or adding additional parameters
        }
        return index_class(**common_params)

    def assert_common_settings(self, settings):
        # Helper method to assert common settings for all index types
        expected_fields = [
            "type", "model", "normalizeEmbeddings", "textPreprocessing",
            "imagePreprocessing", "vectorNumericType", "annParameters"
        ]
        for field in expected_fields:
            self.assertIn(field, settings)

        # Ensure that video and audio preprocessing are not present
        self.assertNotIn("videoPreprocessing", settings)
        self.assertNotIn("audioPreprocessing", settings)

    def test_get_settings_pre_2_12_index(self):
        # Test for a structured index created before version 2.12
        mock_index = self.create_mock_index(
            StructuredMarqoIndex,
            name="test_index",
            type="structured",
            fields=[],
            tensor_fields=[]
        )
        self.mock_config.index_management.get_index.return_value = mock_index

        response = self.client.get("/indexes/test_index/settings")
        self.assertEqual(response.status_code, 200)
        settings = response.json()

        self.assert_common_settings(settings)

    def test_get_settings_pre_2_12_unstructured_index(self):
        # Test for an unstructured index created before version 2.12
        mock_index = self.create_mock_index(
            UnstructuredMarqoIndex,
            name="test_unstructured_index",
            type="unstructured",
            treat_urls_and_pointers_as_images=True,
            filter_string_max_length=200
        )
        self.mock_config.index_management.get_index.return_value = mock_index

        response = self.client.get("/indexes/test_unstructured_index/settings")
        self.assertEqual(response.status_code, 200)
        settings = response.json()

        self.assert_common_settings(settings)
        # Check for unstructured index specific fields
        self.assertIn("treatUrlsAndPointersAsImages", settings)
        self.assertIn("filterStringMaxLength", settings)