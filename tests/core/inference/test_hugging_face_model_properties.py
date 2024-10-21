import unittest
from unittest import mock

from pydantic import ValidationError

from marqo.core.inference.inference_models.hugging_face_model_properties import HuggingFaceModelProperties, PoolingMethod
from marqo.tensor_search.models.external_apis.hf import HfModelLocation
from marqo.tensor_search.models.private_models import ModelLocation


class TestHuggingFaceModelProperties(unittest.TestCase):

    def test_valid_model_with_mandatory_fields(self):
        model = HuggingFaceModelProperties(name="test-model", type="hf", dimensions=768)
        self.assertEqual(model.name, "test-model")
        self.assertEqual(model.token, 128)
        self.assertEqual(model.type, "hf")
        self.assertEqual(model.pooling_method, PoolingMethod.Mean)

    def test_invalid_type(self):
        with self.assertRaises(ValidationError) as excinfo:
            HuggingFaceModelProperties(name="test-model", type="invalid_type", dimensions=768)
        self.assertIn("The type of the model should be 'hf'", str(excinfo.exception))

    def test_valid_model_with_url(self):
        model = HuggingFaceModelProperties(name="test-model", type="hf", url="http://example.com", dimensions=768)
        self.assertEqual(model.url, "http://example.com")
        self.assertIsNone(model.model_location)

    def test_valid_model_with_model_location(self):
        model_location = ModelLocation(hf=HfModelLocation(repo_id="test-repo-id", filename="test-filename"))
        model = HuggingFaceModelProperties(name="test-model", type="hf", model_location=model_location, dimensions=768)
        self.assertEqual(model.model_location, model_location)
        self.assertIsNone(model.url)

    def test_invalid_model_with_url_and_model_location(self):
        with self.assertRaises(ValidationError) as excinfo:
            HuggingFaceModelProperties(
                name="test-model", type="hf",
                url="http://example.com",
                model_location=ModelLocation(hf=HfModelLocation(repo_id="test-repo-id", filename="test-filename")),
                dimensions=768
            )
        self.assertIn("Only one of 'url' and 'model_location' should be provided.", str(excinfo.exception))

    def test_infer_pooling_method(self):
        for pooling_method in (PoolingMethod.Mean, PoolingMethod.CLS):
            with self.subTest(f"Pooling method inferred from name with {pooling_method}"):
                with mock.patch("marqo.core.inference.inference_models.hugging_face_model_properties."
                                "HuggingFaceModelProperties._infer_pooling_method_from_name",
                                return_value = pooling_method) as mock_infer:
                    model = HuggingFaceModelProperties(name="model-with-cls", type="hf", dimensions=768)
                mock_infer.assert_called_once()
                self.assertEqual(pooling_method, model.pooling_method)

    def test_explicit_valid_pooling_method(self):
        with mock.patch("marqo.core.inference.inference_models.hugging_face_model_properties."
                        "HuggingFaceModelProperties._infer_pooling_method_from_name") as mock_infer:
            model = HuggingFaceModelProperties(name="test-model", type="hf", pooling_method=PoolingMethod.CLS,
                                               dimensions=768)
        self.assertEqual(model.pooling_method, PoolingMethod.CLS)
        mock_infer.assert_not_called()

    def test_explicit_invalid_pooling_method(self):
        with self.assertRaises(ValidationError) as excinfo:
            _ = HuggingFaceModelProperties(name="test-model", type="hf", pooling_method="invalid", dimensions=768)
        self.assertIn("value is not a valid enumeration member; permitted: 'mean', 'cls'",
                      str(excinfo.exception))

    def test_model_without_optional_fields(self):
        model = HuggingFaceModelProperties(name="test-model", type="hf", dimensions=768)
        self.assertIsNone(model.url)
        self.assertIsNone(model.model_location)
        self.assertIsNone(model.model_auth)
        self.assertIsNone(model.note)
        self.assertEqual(model.pooling_method, PoolingMethod.Mean)

    def test_invalid_model_without_minimum_fields(self):
        with self.assertRaises(ValidationError) as excinfo:
            HuggingFaceModelProperties(type="hf", dimensions=768)
        self.assertIn("At least one of 'name', 'url', or 'model_location' should be provided.", str(excinfo.exception))

    def test_invalid_model_with_both_url_and_model_location(self):
        model_location = ModelLocation(hf=HfModelLocation(repo_id="test-repo-id", filename="test-filename"))
        with self.assertRaises(ValidationError) as excinfo:
            HuggingFaceModelProperties(url="http://example.com", model_location=model_location, type="hf", dimensions=768)
        self.assertIn("Only one of 'url' and 'model_location' should be provided.", str(excinfo.exception))

    def test_valid_model_with_custom_url_and_inferred_pooling(self):
        model = HuggingFaceModelProperties(url="http://example.com", type="hf", pooling_method=None, dimensions=768)
        self.assertEqual(model.pooling_method, PoolingMethod.Mean)

    def test_some_pooling_method_infer_on_real_model(self):
        test_cases = [
            ("intfloat/e5-base-v2", PoolingMethod.Mean),
            ("sentence-transformers/all-MiniLM-L6-v2", PoolingMethod.Mean),
            ("sentence-transformers/paraphrase-MiniLM-L3-v2", PoolingMethod.Mean),
            ("sentence-transformers/all-mpnet-base-v2", PoolingMethod.Mean),
            ("sentence-transformers/bert-base-nli-mean-tokens", PoolingMethod.Mean),

            ("sentence-transformers/nli-bert-base-cls-pooling", PoolingMethod.CLS),
            ("sentence-transformers/nli-bert-large-cls-pooling", PoolingMethod.CLS),
        ]

        for model_name, pooling_method in test_cases:
            with self.subTest(f"Pooling method inferred from name with {model_name}"):
                model = HuggingFaceModelProperties(name=model_name, type="hf", dimensions=768)
                self.assertEqual(pooling_method, model.pooling_method)