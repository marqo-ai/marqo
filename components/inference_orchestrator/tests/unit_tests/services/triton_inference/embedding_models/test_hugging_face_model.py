import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.errors import (
    InternalServerError,
    InvalidModelPropertiesError,
)
from inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model import (
    HuggingFaceModel,
    HuggingFacePreprocessor,
)
from inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model_properties import (
    PoolingMethod,
)


class TestHuggingFacePreprocessor(unittest.TestCase):
    """Test suite for HuggingFacePreprocessor class.

    These tests verify the preprocessing functionality for HuggingFace models.
    HuggingFace models don't require preprocessing - inputs are passed through as-is.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = HuggingFacePreprocessor()

    def test_preprocess_text_returns_input_unchanged(self):
        """Test that preprocessing returns input strings unchanged."""
        inputs = ["hello world", "test text", "another string"]

        result = self.preprocessor.preprocess(inputs, Modality.TEXT)

        self.assertEqual(inputs, result)

    def test_preprocess_with_single_input(self):
        """Test preprocessing with a single input."""
        inputs = ["single text"]

        result = self.preprocessor.preprocess(inputs, Modality.TEXT)

        self.assertEqual(inputs, result)

    def test_preprocess_with_empty_list(self):
        """Test preprocessing with an empty list."""
        inputs = []

        result = self.preprocessor.preprocess(inputs, Modality.TEXT)

        self.assertEqual(inputs, result)

    def test_preprocess_modality_parameter_does_not_affect_output(self):
        """Test that modality parameter doesn't affect output (HF models only support TEXT)."""
        inputs = ["test"]

        # Should return same result regardless of modality
        result = self.preprocessor.preprocess(inputs, Modality.TEXT)

        self.assertEqual(inputs, result)


class TestHuggingFaceModel(unittest.TestCase):
    """Test suite for HuggingFaceModel class.

    These tests verify the HuggingFaceModel functionality without loading real models.
    All external dependencies (transformers, triton) are mocked.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model_management_client = MagicMock()
        self.mock_triton_client = MagicMock()

        # Valid model properties for testing
        self.valid_model_properties = {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "hf",
            "dimensions": 384,
            "tokens": 128,
            "poolingMethod": "mean",
            "tritonTextEncoderProperties": {
                "name": "text-encoder",
                "sources": ["s3://bucket/text-encoder/model.onnx"],
                "input": [
                    {"name": "input_ids", "dims": [1, 128], "dataType": "TYPE_INT64"},
                    {
                        "name": "attention_mask",
                        "dims": [1, 128],
                        "dataType": "TYPE_INT64",
                    },
                    {
                        "name": "token_type_ids",
                        "dims": [1, 128],
                        "dataType": "TYPE_INT64",
                    },
                ],
                "output": [
                    {
                        "name": "last_hidden_state",
                        "dims": [1, 128, 384],
                        "dataType": "TYPE_FP32",
                    }
                ],
                "maxBatchSize": 32,
            },
        }

        self.valid_cls_model_properties = {
            "name": "bert-base-uncased",
            "type": "hf",
            "dimensions": 768,
            "tokens": 512,
            "poolingMethod": "cls",
            "tritonTextEncoderProperties": {
                "name": "bert-encoder",
                "sources": ["s3://bucket/bert/model.onnx"],
                "input": [
                    {"name": "input_ids", "dims": [1, 512], "dataType": "TYPE_INT64"},
                    {
                        "name": "attention_mask",
                        "dims": [1, 512],
                        "dataType": "TYPE_INT64",
                    },
                    {
                        "name": "token_type_ids",
                        "dims": [1, 512],
                        "dataType": "TYPE_INT64",
                    },
                ],
                "output": [
                    {
                        "name": "last_hidden_state",
                        "dims": [1, 512, 768],
                        "dataType": "TYPE_FP32",
                    }
                ],
                "maxBatchSize": 16,
            },
        }

    def test_init_with_valid_properties(self):
        """Test HuggingFaceModel initialization with valid properties."""
        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        self.assertIsNotNone(model.model_properties)
        self.assertEqual(
            "sentence-transformers/all-MiniLM-L6-v2", model.model_properties.name
        )
        self.assertEqual(384, model.model_properties.dimensions)
        self.assertEqual(PoolingMethod.Mean, model.model_properties.pooling_method)

    def test_init_with_cls_pooling(self):
        """Test initialization with CLS pooling method."""
        model = HuggingFaceModel(
            model_properties=self.valid_cls_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        self.assertEqual(PoolingMethod.CLS, model.model_properties.pooling_method)
        self.assertEqual(768, model.model_properties.dimensions)

    def test_init_with_invalid_properties_raises_error(self):
        """Test that invalid model properties raise InvalidModelPropertiesError."""
        invalid_properties = {
            "name": "test-model",
            "type": "wrong_type",  # Invalid type
            "dimensions": 384,
        }

        with self.assertRaises(InvalidModelPropertiesError) as context:
            HuggingFaceModel(
                model_properties=invalid_properties,
                model_management_client=self.mock_model_management_client,
                triton_client=self.mock_triton_client,
            )

        self.assertIn("Invalid model properties", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_load_with_mean_pooling(self, mock_auto_tokenizer):
        """Test loading model with mean pooling method."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        # Execute load
        model.load()

        # Verify tokenizer was loaded
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_args = mock_auto_tokenizer.from_pretrained.call_args[0]
        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", call_args[0])

        # Verify pooling function was set
        self.assertIsNotNone(model._pooling_func)
        self.assertEqual(model._average_pool_func, model._pooling_func)

        # Verify triton model was loaded
        self.mock_model_management_client.load_model.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_load_with_cls_pooling(self, mock_auto_tokenizer):
        """Test loading model with CLS pooling method."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model = HuggingFaceModel(
            model_properties=self.valid_cls_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        # Execute load
        model.load()

        # Verify pooling function is CLS
        self.assertEqual(model._cls_pool_func, model._pooling_func)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_load_uses_effective_name_from_triton_model_name(self, mock_auto_tokenizer):
        """Test that loading uses tritonModelName (via effective_name) when set.

        During migration, the Vespa-stored properties keep the old 'name' for cache key stability
        and add 'tritonModelName' with the new value for the inference orchestrator.
        """
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Properties with old name but new tritonModelName
        properties_with_triton_name = self.valid_model_properties.copy()
        properties_with_triton_name["name"] = "old-model-name"
        properties_with_triton_name["tritonModelName"] = "sentence-transformers/all-MiniLM-L6-v2"

        model = HuggingFaceModel(
            model_properties=properties_with_triton_name,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model.load()

        # Verify AutoTokenizer.from_pretrained used the tritonModelName (effective_name)
        call_args = mock_auto_tokenizer.from_pretrained.call_args[0]
        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", call_args[0])

        # Verify the original name is preserved
        self.assertEqual("old-model-name", model.model_properties.name)

    def test_check_loaded_components_raises_error_if_tokenizer_not_loaded(self):
        """Test that _check_loaded_components raises error if tokenizer is None."""
        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model._tokenizer = None
        model._pooling_func = MagicMock()

        with self.assertRaises(InternalServerError) as context:
            model._check_loaded_components()

        self.assertIn("tokenizer is not loaded", str(context.exception).lower())

    def test_check_loaded_components_raises_error_if_pooling_func_not_loaded(self):
        """Test that _check_loaded_components raises error if pooling_func is None."""
        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        model._tokenizer = MagicMock()
        model._pooling_func = None

        with self.assertRaises(InternalServerError) as context:
            model._check_loaded_components()

        self.assertIn("pooling function is not loaded", str(context.exception).lower())

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_get_preprocessor_returns_huggingface_preprocessor(
        self, mock_auto_tokenizer
    ):
        """Test that get_preprocessor returns HuggingFacePreprocessor."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

        preprocessor = model.get_preprocessor()

        self.assertIsInstance(preprocessor, HuggingFacePreprocessor)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_encode_with_mean_pooling(self, mock_auto_tokenizer):
        """Test encoding with mean pooling."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": np.array([[101, 2023, 2003, 102]]),
            "attention_mask": np.array([[1, 1, 1, 1]]),
            "token_type_ids": np.array([[0, 0, 0, 0]]),
        }
        mock_tokenizer.return_value = mock_encoded
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup mock triton response
        mock_response = MagicMock()
        # Shape: (batch_size, seq_len, hidden_dim) = (1, 4, 384)
        last_hidden_state = np.random.rand(1, 4, 384).astype(np.float32)
        mock_response.as_numpy.return_value = last_hidden_state
        self.mock_triton_client.encode.return_value = mock_response

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute
        inputs = ["test sentence"]
        result = model.encode(inputs, modality=Modality.TEXT, normalize=True)

        # Verify tokenizer was called
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        self.assertTrue(call_kwargs["padding"])
        self.assertTrue(call_kwargs["truncation"])
        self.assertEqual(128, call_kwargs["max_length"])

        # Verify triton client was called
        self.mock_triton_client.encode.assert_called_once()
        call_kwargs = self.mock_triton_client.encode.call_args[1]
        self.assertEqual("text-encoder", call_kwargs["model_name"])

        # Verify result
        self.assertEqual(1, len(result))
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(384, len(result[0]))

        # Verify normalization was applied
        norm = np.linalg.norm(result[0])
        self.assertAlmostEqual(1.0, norm, places=5)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_encode_with_cls_pooling(self, mock_auto_tokenizer):
        """Test encoding with CLS pooling."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": np.array([[101, 2023, 102]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "token_type_ids": np.array([[0, 0, 0]]),
        }
        mock_tokenizer.return_value = mock_encoded
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup mock triton response
        mock_response = MagicMock()
        # Shape: (batch_size, seq_len, hidden_dim) = (1, 3, 768)
        last_hidden_state = np.random.rand(1, 3, 768).astype(np.float32)
        mock_response.as_numpy.return_value = last_hidden_state
        self.mock_triton_client.encode.return_value = mock_response

        model = HuggingFaceModel(
            model_properties=self.valid_cls_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute
        inputs = ["test"]
        result = model.encode(inputs, modality=Modality.TEXT, normalize=True)

        # Verify result
        self.assertEqual(1, len(result))
        self.assertEqual(768, len(result[0]))

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_encode_without_normalization(self, mock_auto_tokenizer):
        """Test encoding without normalization."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": np.array([[101, 2023, 102]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "token_type_ids": np.array([[0, 0, 0]]),
        }
        mock_tokenizer.return_value = mock_encoded
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup mock triton response with unnormalized embeddings
        mock_response = MagicMock()
        last_hidden_state = np.array([[[3.0] * 384, [2.0] * 384, [1.0] * 384]])
        mock_response.as_numpy.return_value = last_hidden_state
        self.mock_triton_client.encode.return_value = mock_response

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute without normalization
        result = model.encode(["test"], modality=Modality.TEXT, normalize=False)

        # Verify result is not normalized (norm should be > 1)
        norm = np.linalg.norm(result[0])
        self.assertGreater(norm, 1.0)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_encode_multiple_inputs(self, mock_auto_tokenizer):
        """Test encoding multiple inputs in a batch."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": np.array([[101, 2023, 102], [101, 2054, 102]]),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
            "token_type_ids": np.array([[0, 0, 0], [0, 0, 0]]),
        }
        mock_tokenizer.return_value = mock_encoded
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup mock triton response
        mock_response = MagicMock()
        last_hidden_state = np.random.rand(2, 3, 384).astype(np.float32)
        mock_response.as_numpy.return_value = last_hidden_state
        self.mock_triton_client.encode.return_value = mock_response

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute
        inputs = ["first sentence", "second sentence"]
        result = model.encode(inputs, modality=Modality.TEXT, normalize=True)

        # Verify result
        self.assertEqual(2, len(result))
        for embedding in result:
            self.assertEqual(384, len(embedding))
            # Check normalization
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(1.0, norm, places=5)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_encode_with_invalid_input_raises_error(self, mock_auto_tokenizer):
        """Test that encoding with invalid input raises InternalServerError."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Test with non-list input
        with self.assertRaises(InternalServerError) as context:
            model.encode("not a list", modality=Modality.TEXT, normalize=True)

        self.assertIn("list of strings", str(context.exception).lower())

    def test_average_pool_func(self):
        """Test the mean pooling function."""
        # Create test data
        # Shape: (batch_size, seq_len, hidden_dim) = (2, 3, 4)
        model_output = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]],
            ]
        )
        attention_mask = np.array([[1, 1, 0], [1, 1, 1]])  # Second item has padding

        result = HuggingFaceModel._average_pool_func(model_output, attention_mask)

        # Verify shape
        self.assertEqual((2, 4), result.shape)

        # First sequence: average of first 2 tokens (third is masked)
        expected_first = np.mean(model_output[0, :2, :], axis=0)
        np.testing.assert_array_almost_equal(expected_first, result[0])

        # Second sequence: average of all 3 tokens
        expected_second = np.mean(model_output[1, :, :], axis=0)
        np.testing.assert_array_almost_equal(expected_second, result[1])

    def test_cls_pool_func(self):
        """Test the CLS pooling function."""
        # Create test data
        # Shape: (batch_size, seq_len, hidden_dim) = (2, 3, 4)
        model_output = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]],
            ]
        )
        attention_mask = np.array([[1, 1, 1], [1, 1, 1]])

        result = HuggingFaceModel._cls_pool_func(model_output, attention_mask)

        # Verify shape
        self.assertEqual((2, 4), result.shape)

        # Should return first token (CLS token) for each sequence
        np.testing.assert_array_equal(model_output[:, 0, :], result)

    @patch(
        "inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model.AutoTokenizer"
    )
    def test_unload_calls_model_management_client(self, mock_auto_tokenizer):
        """Test that unload calls model_management_client."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model = HuggingFaceModel(
            model_properties=self.valid_model_properties,
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )
        model.load()

        # Execute unload
        model.unload(remove_files=True)

        # Verify unload_model was called
        self.mock_model_management_client.unload_model.assert_called_once()
        call_args = self.mock_model_management_client.unload_model.call_args
        self.assertEqual("text-encoder", call_args[0][0])
        self.assertEqual(True, call_args[1]["remove_files"])

    def test_average_pool_func_with_all_masked(self):
        """Test mean pooling with edge case where some tokens might be masked."""
        # Create test data where one sequence is fully masked (edge case)
        model_output = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = np.array([[0, 0, 0]])  # All masked

        result = HuggingFaceModel._average_pool_func(model_output, attention_mask)

        # Verify shape is correct
        self.assertEqual((1, 2), result.shape)

        # Should not divide by zero (clipped to 1e-9)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


if __name__ == "__main__":
    unittest.main()
