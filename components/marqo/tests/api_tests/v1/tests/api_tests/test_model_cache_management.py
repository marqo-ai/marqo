import unittest
import uuid

import pytest
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase
import requests


class TestModlCacheManagement(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text"},
                ],
                "tensorFields": ["title"]
            },
            {
                "indexName": cls.unstructured_index_name,
                "model": "hf/all-MiniLM-L6-v2",
                "type": "unstructured",
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def setUp(self):
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            # Do a search to load the model into cache
            self.client.index(index_name).search("test")

    def test_get_cpu_info(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                r = self.client.index(index_name).get_cpu_info()
                self.assertIn("cpu_usage_percent", r)
                self.assertIn("memory_used_percent", r)
                self.assertIn("memory_used_gb", r)

    def test_get_loaded_models_format(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                loaded_models :list[dict] = requests.get(f"{self._MARQO_URL}/models?detailed=true").json()
                self.assertIn("models", loaded_models)
                models = loaded_models["models"]
                for model in models:
                    self.assertIn("modelName", model)
                    self.assertIn("modelProperties", model)

    def test_get_loaded_models_format_detailed_false(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                loaded_models :list[dict] = requests.get(f"{self._MARQO_URL}/models?detailed=false").json()
                self.assertIn("models", loaded_models)
                models = loaded_models["models"]
                for model in models:
                    self.assertIn("modelName", model)
                    self.assertNotIn("modelProperties", model)

    def test_get_loaded_models_format_detailed_default(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                loaded_models :list[dict] = requests.get(f"{self._MARQO_URL}/models").json()
                self.assertIn("models", loaded_models)
                models = loaded_models["models"]
                for model in models:
                    self.assertIn("modelName", model)
                    self.assertNotIn("modelProperties", model)

    def test_eject_model(self) -> None:
        # test eject a model that is cached
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                # Do a search to ensure the model is cached
                r = self.client.index(index_name).search("q")

                loaded_models = self.client.index(index_name).get_loaded_models()["models"]
                for model in loaded_models:
                    res = requests.delete(f"{self._MARQO_URL}/models?model_name={model['modelName']}").json()
                    self.assertIn("ejected successfully", str(res))
