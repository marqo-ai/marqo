import uuid

from marqo.core.models.marqo_index import *
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestModelManagerClient(MarqoTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clear_all_loaded_models()

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k")
        )
        cls.index_image_name = unstructured_image_index_request.name

        unstructured_text_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_text_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/all-MiniLM-L6-v2")
        )
        cls.index_text_name = unstructured_text_index_request.name

        cls.create_indexes(
            [unstructured_image_index_request, unstructured_text_index_request]
        )

        cls.indexes_to_cleanup = [unstructured_image_index_request.name, unstructured_text_index_request.name]

    def setUp(self):
        # Do a simple search to ensure the model is loaded before running tests
        super().setUp()
        for index_name in self.indexes_to_cleanup:
            tensor_search.search(
                config=self.config, index_name=index_name, text="test",
                search_method=SearchMethod.TENSOR
            )

    def test_get_loaded_models_default_detailed(self):
        loaded_models = self.config.model_manager.get_loaded_models()
        self.assertIsInstance(loaded_models, dict)
        self.assertIn("models", loaded_models)
        self.assertIsInstance(loaded_models["models"], list)
        self.assertEqual(2, len(loaded_models["models"]))

        models = loaded_models["models"]

        for model in models:
            self.assertIn("modelName", model)
            self.assertNotIn("device", model)
            self.assertNotIn("modelProperties", model)

    def test_get_loaded_models_detailed_false(self):
        loaded_models = self.config.model_manager.get_loaded_models(detailed=False)
        self.assertIsInstance(loaded_models, dict)
        self.assertIn("models", loaded_models)
        self.assertIsInstance(loaded_models["models"], list)
        self.assertEqual(2, len(loaded_models["models"]))

        models = loaded_models["models"]

        for model in models:
            self.assertIn("modelName", model)
            self.assertNotIn("device", model)
            self.assertNotIn("modelProperties", model)

    def test_get_loaded_models_detailed_true(self):
        loaded_models = self.config.model_manager.get_loaded_models(detailed=True)
        self.assertIsInstance(loaded_models, dict)
        self.assertIn("models", loaded_models)
        self.assertIsInstance(loaded_models["models"], list)
        self.assertEqual(2, len(loaded_models["models"]))
        models = loaded_models["models"]
        for model in models:
            self.assertIn("modelName", model)
            self.assertNotIn("device", model)
            self.assertIn("modelProperties", model)