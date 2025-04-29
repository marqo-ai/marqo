from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.inference.native_inference.inference_pipeline.languagebind_model_inference_pipeline import (
    LanguagebindModelInferencePipeline
)
from marqo.inference.type import *


class TestLanguagebindModelInferencePipeline(TestCase):
    """
    Unittests for LanguagebindModelInferencePipeline. This should not include any real downloading or preprocessing.
    """

    def setUp(self):
        self.mock_model = Mock(spec=LanguagebindModel)

        # Mock get_preprocessor to return a callable
        # That callable returns a list of Tensor mocks
        tensor_mock = Mock(spec=Tensor)
        preprocessor_callable = Mock(return_value=[tensor_mock])  # returns list when called
        self.mock_model.get_preprocessor.return_value = preprocessor_callable

        self.mock_model.encode.return_value = [np.random.rand(768)]

        # Mock model_properties
        mock_model_properties = Mock()
        mock_model_properties.supportedModalities = ["language", "image", "audio", "video"]
        self.mock_model.model_properties = mock_model_properties

        # Valid configs
        self.model_config = ModelConfig(
            model_name="mock-model",
            model_properties={"name": "mock-model", "dimensions": 768, "supported_modalities": ["language", "image"]},
            normalize_embeddings=True
        )

        self.preprocessing_config = TextPreprocessingConfig(
            should_chunk=False
        )

        self.inference_request_text = InferenceRequest(
            modality=Modality.TEXT,
            contents=["sample text"],
            model_config=self.model_config,
            preprocessing_config=self.preprocessing_config,
            return_individual_error=False,
            device="cpu"
        )

        self.pipeline = LanguagebindModelInferencePipeline(
            model=self.mock_model,
            inference_request=self.inference_request_text
        )

    def test_collect_valid_content_to_encode_valid(self):
        """Test to ensure that the method collects valid content to encode."""
        tensor_mock = Mock(spec=Tensor)
        preprocessed_content = [[("chunk", tensor_mock)]]

        result = self.pipeline._collect_valid_content_to_encode(preprocessed_content)
        self.assertEqual(result, [tensor_mock])

    def test_collect_valid_content_to_encode_with_error_model(self):
        """Ensure the method skip the error model."""
        error_model = InferenceErrorModel(error_message="error")
        preprocessed_content = [error_model]

        result = self.pipeline._collect_valid_content_to_encode(preprocessed_content)
        self.assertEqual(result, [])

    def test_collect_valid_content_to_encode_invalid_content_type(self):
        """Ensure an internal error is raised when the content type is invalid."""
        preprocessed_content = [["invalid_string"]]
        with self.assertRaises(ValueError):
            self.pipeline._collect_valid_content_to_encode(preprocessed_content)

    def test_encode_processed_content_valid(self):
        """Test to ensure that the method encodes the processed content correctly."""
        tensor_mock = Mock(spec=Tensor)
        preprocessed_content = [[("chunk", tensor_mock)]]

        result = self.pipeline._encode_processed_content(preprocessed_content)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)

    def test_encode_processed_content_empty(self):
        """Test to ensure that the method handles empty content correctly."""
        result = self.pipeline._encode_processed_content([])
        self.assertEqual(result, [])

    def test_encode_processed_content_mismatch_embeddings(self):
        """Test to ensure that the method raises an error when the number of
        embeddings does not match the number of contents."""
        tensor_mock = Mock(spec=Tensor)
        preprocessed_content = [[("chunk", tensor_mock), ("chunk", tensor_mock)]]
        # Force only one embedding to simulate mismatch
        self.mock_model.encode.return_value = [np.random.rand(768)]

        with self.assertRaises(ValueError):
            self.pipeline._encode_processed_content(preprocessed_content)

    @patch.object(LanguagebindModelInferencePipeline, '_content_preprocessing')
    @patch.object(LanguagebindModelInferencePipeline, '_encode_processed_content')
    @patch.object(LanguagebindModelInferencePipeline, 'format_results')
    def test_run_pipeline(self, mock_format_results, mock_encode, mock_preprocess):
        """Test to ensure that the pipeline runs correctly."""
        mock_preprocess.return_value = [[("chunk", Mock(spec=Tensor))]]
        mock_encode.return_value = [np.random.rand(768)]
        mock_format_results.return_value = InferenceResult(result=[[]])

        result = self.pipeline.run_pipeline()

        mock_preprocess.assert_called_once()
        mock_encode.assert_called_once()
        mock_format_results.assert_called_once()
        self.assertIsInstance(result, InferenceResult)
