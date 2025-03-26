import unittest

import numpy as np
from pydantic import ValidationError

from marqo.core.inference.api import ModelConfig, InferenceRequest, Modality, TextPreprocessingConfig, \
    AudioPreprocessingConfig, VideoPreprocessingConfig, ImagePreprocessingConfig, InferenceResult, \
    InferenceErrorModel
from marqo.tensor_search.models.external_apis.hf import HfAuth
from marqo.tensor_search.models.private_models import ModelAuth


class TestModelConfig(unittest.TestCase):
    def test_valid_model_config(self):
        """Test creating ModelConfig with all valid fields."""
        auth = ModelAuth(hf=HfAuth(token='<PASSWORD>'))
        config = ModelConfig(
            model_name="test_model",
            model_properties={"property1": "value1"},
            model_auth=auth,
            normalize_embeddings=False
        )
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.model_properties, {"property1": "value1"})
        self.assertEqual(config.model_auth, auth)
        self.assertFalse(config.normalize_embeddings)

    def test_default_values(self):
        """Test creating ModelConfig with only required fields."""
        config = ModelConfig(
            model_name="test_model"
        )
        self.assertEqual(config.model_name, "test_model")
        self.assertIsNone(config.model_properties)
        self.assertIsNone(config.model_auth)
        self.assertTrue(config.normalize_embeddings)

    def test_missing_required_field(self):
        """Test that missing a required field raises a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ModelConfig(
                model_properties={"property1": "value1"},
                normalize_embeddings=True
            )
        self.assertIn("modelName", str(context.exception))
        self.assertIn("field required (type=value_error.missing)", str(context.exception))

    def test_invalid_model_name_type(self):
        """Test that providing a non-string model_name raises a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ModelConfig(
                model_name=123,  # Invalid type
                model_properties={"property1": "value1"},
                normalize_embeddings=True
            )
        self.assertIn("modelName", str(context.exception))
        self.assertIn("str type expected (type=type_error.str)", str(context.exception))

    def test_invalid_model_properties_type(self):
        """Test that providing a non-dict model_properties raises a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ModelConfig(
                model_name="test_model",
                model_properties="not_a_dict",  # Invalid type
                normalize_embeddings=True
            )
        self.assertIn("modelProperties", str(context.exception))

    def test_alias_usage(self):
        """Test that aliases are correctly interpreted."""
        config = ModelConfig(
            modelName="test_model_alias",
            modelProperties={"property1": "value1"},
            modelAuth={"hf": {"token": "test_token"}},
            normalizeEmbeddings=False
        )
        self.assertEqual(config.model_name, "test_model_alias")
        self.assertIsInstance(config.model_auth, ModelAuth)
        self.assertEqual(config.model_auth.hf.token, "test_token")
        self.assertFalse(config.normalize_embeddings)
        self.assertEqual(config.model_properties, {"property1": "value1"})

    def test_immutability(self):
        """Test that the ModelConfig instance is immutable."""
        config = ModelConfig(
            modelName="immutable_model",
            normalizeEmbeddings=True
        )
        with self.assertRaises(TypeError) as context:
            config.model_name = "new_model_name"
        self.assertIn('"ModelConfig" is immutable and does not support item assignment', str(context.exception))


class TestInferenceRequest(unittest.TestCase):

    def setUp(self):
        self.model_config = ModelConfig(model_name="test_model")

    def test_empty_contents(self):
        """Test that empty contents list raises a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            InferenceRequest(
                modality=Modality.TEXT,
                contents=[],
                model_config=self.model_config,
                preprocessing_config=TextPreprocessingConfig()
            )
        self.assertIn('ensure this value has at least 1 items', str(context.exception))

    def test_invalid_contents_type(self):
        """Test that non-list contents raise a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            InferenceRequest(
                modality=Modality.VIDEO,
                contents="This should be a list",
                model_config=self.model_config,
                preprocessing_config=TextPreprocessingConfig()
            )
        self.assertIn('value is not a valid list', str(context.exception))

    def test_invalid_modality(self):
        """Test that an invalid modality raises a ValidationError."""
        with self.assertRaises(ValidationError) as context:
            InferenceRequest(
                modality="INVALID_MODALITY",
                contents=["Sample text"],
                model_config=self.model_config,
                preprocessing_config=TextPreprocessingConfig()
            )
        self.assertIn('value is not a valid enumeration member', str(context.exception))

    def test_valid_inference_request_for_matching_modality(self):
        """Test creating a valid InferenceRequest with all required fields."""
        for modality, preprocessing_config in [
            (Modality.TEXT, TextPreprocessingConfig()),
            (Modality.AUDIO, AudioPreprocessingConfig()),
            (Modality.VIDEO, VideoPreprocessingConfig()),
            (Modality.IMAGE, ImagePreprocessingConfig()),
        ]:
            with self.subTest(modality=modality, preprocessing_config=preprocessing_config):
                request = InferenceRequest(
                    modality=modality,
                    contents=["some content"],
                    model_config=self.model_config,
                    preprocessing_config=preprocessing_config
                )
                self.assertEqual(request.modality, modality)
                self.assertEqual(request.preprocessing_config, preprocessing_config)

    def test_invalid_inference_request_with_missing_modality(self):
        with self.assertRaises(ValidationError) as context:
            InferenceRequest(
                modality=None,
                contents=["some content"],
                model_config=self.model_config,
                preprocessing_config=TextPreprocessingConfig()
            )
        self.assertIn('Modality or preprocessing_config is missing', str(context.exception))

    def test_invalid_inference_request_with_missing_preprocessing_config(self):
        with self.assertRaises(ValidationError) as context:
            InferenceRequest(
                modality=Modality.TEXT,
                contents=["some content"],
                model_config=self.model_config,
                preprocessing_config=None
            )
        self.assertIn('Modality or preprocessing_config is missing', str(context.exception))

    def test_invalid_inference_request_for_non_matching_modality(self):
        for modality, preprocessing_config in [
            (Modality.TEXT, ImagePreprocessingConfig()),
            (Modality.TEXT, AudioPreprocessingConfig()),
            (Modality.TEXT, VideoPreprocessingConfig()),

            (Modality.IMAGE, TextPreprocessingConfig()),
            (Modality.IMAGE, AudioPreprocessingConfig()),
            (Modality.IMAGE, VideoPreprocessingConfig()),

            (Modality.AUDIO, TextPreprocessingConfig()),
            (Modality.AUDIO, ImagePreprocessingConfig()),
            (Modality.AUDIO, VideoPreprocessingConfig()),

            (Modality.VIDEO, TextPreprocessingConfig()),
            (Modality.VIDEO, ImagePreprocessingConfig()),
            (Modality.VIDEO, AudioPreprocessingConfig()),
        ]:

            with self.subTest(modality=modality, preprocessing_config=preprocessing_config):
                with self.assertRaises(ValidationError) as context:
                    InferenceRequest(
                        modality=modality,
                        contents=["some content"],
                        model_config=self.model_config,
                        preprocessing_config=preprocessing_config
                    )
                self.assertIn('does not support modality:', str(context.exception))

    def test_default_values(self):
        """Test that default values are set correctly when optional fields are not provided."""
        request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=["image1.png", "image2.png"],
            model_config=self.model_config,
            preprocessing_config=ImagePreprocessingConfig()
        )
        self.assertIsNone(request.device)
        self.assertFalse(request.use_inference_cache)
        self.assertTrue(request.return_individual_error)

    def test_custom_device_and_use_inference_cache(self):
        """Test setting custom values for optional fields."""
        request = InferenceRequest(
            modality=Modality.AUDIO,
            contents=["audio1.mp3"],
            device="cuda",
            use_inference_cache=True,
            model_config=self.model_config,
            preprocessing_config=AudioPreprocessingConfig(),
            return_individual_error=False,
        )
        self.assertEqual(request.device, "cuda")
        self.assertTrue(request.use_inference_cache)
        self.assertFalse(request.return_individual_error)

    def test_alias_fields(self):
        """Test that alias fields are correctly handled."""
        data = {
            "modality": "language",
            "contents": ["Sample text"],
            "modelConfig": self.model_config,
            "preprocessingConfig": TextPreprocessingConfig(),
            "useInferenceCache": True
        }
        request = InferenceRequest(**data)
        self.assertEqual(request.modality, Modality.TEXT)
        self.assertTrue(request.use_inference_cache)
        self.assertEqual(request.model_config, self.model_config)

    def test_immutability(self):
        request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=["image1.png", "image2.png"],
            model_config=self.model_config,
            preprocessing_config=ImagePreprocessingConfig()
        )

        with self.assertRaises(TypeError) as context:
            request.contents = ["other content"]

        self.assertIn('"InferenceRequest" is immutable and does not support item assignment', str(context.exception))


class TestInferenceResult(unittest.TestCase):
    def test_inference_error(self):
        error = InferenceErrorModel(error_message="An error occurred")
        result = InferenceResult(result=[error])
        self.assertEqual(len(result.result), 1)
        self.assertIsInstance(result.result[0], InferenceErrorModel)
        self.assertEqual(result.result[0].error_message, "An error occurred")
        self.assertEqual(result.result[0].error_code, "inference_error")
        self.assertEqual(result.result[0].status_code, 400)

    def test_valid_success_result(self):
        data = [("item1", np.array([1, 2, 3])), ("item2", np.array([4, 5, 6]))]
        result = InferenceResult(result=[data])
        self.assertEqual(len(result.result), 1)
        self.assertIsInstance(result.result[0], list)
        self.assertEqual(len(result.result[0]), 2)
        self.assertEqual(result.result[0][0][0], "item1")
        self.assertIsInstance(result.result[0][0][1], np.ndarray)
        np.testing.assert_array_equal(result.result[0][0][1], np.array([1, 2, 3]))

    def test_mixed_results(self):
        error = InferenceErrorModel(error_message="Partial error")
        data = [("item3", np.array([7, 8, 9]))]
        result = InferenceResult(result=[error, data])
        self.assertEqual(len(result.result), 2)
        self.assertIsInstance(result.result[0], InferenceErrorModel)
        self.assertIsInstance(result.result[1], list)
        self.assertEqual(result.result[1][0][0], "item3")
        self.assertIsInstance(result.result[1][0][1], np.ndarray)
        np.testing.assert_array_equal(result.result[1][0][1], np.array([7, 8, 9]))

    def test_invalid_result_type_not_list(self):
        with self.assertRaises(ValidationError):
            InferenceResult(result="this should be a list")

    def test_invalid_union_type(self):
        with self.assertRaises(ValidationError):
            InferenceResult(result=[123])  # Invalid type, neither InferenceErrorModel nor list of tuples

    def test_invalid_tuple_structure(self):
        with self.assertRaises(ValidationError):
            # Second element of tuple should be numpy.ndarray
            InferenceResult(result=[[("item4", "not an ndarray")]])

    def test_immutability(self):
        result = InferenceResult(result=[])
        with self.assertRaises(TypeError):
            result.result = [[("item3", np.array([7, 8, 9]))]]  # Should be immutable
            # Please note that we don't support deep immutability
