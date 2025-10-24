import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.errors import InternalServerError
from inference_orchestrator.services.triton_inference.inference_pipelines import (
    HuggingFaceModelInferencePipeline,
)


class TestHuggingFaceModelInferencePipeline(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.get_preprocessor.return_value = MagicMock()

        # Mock model properties for batching
        self.mock_model.model_properties.triton_text_encoder_properties.max_batch_size = 32

        # Create a real EmbeddingModelConfig for Pydantic validation
        self.embedding_model_config = EmbeddingModelConfig(
            model_name="test-model", normalize_embeddings=True
        )

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_content_preprocessing_text_modality(self, mock_split_preprocess):
        """Ensure that the content preprocessing is done correctly for text modalities."""
        inference_request = InferenceRequest(
            contents=["this is a test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)
        mock_split_preprocess.return_value = [["original", "preprocessed"]]

        result = pipeline._content_preprocessing()

        mock_split_preprocess.assert_called_once_with(
            inference_request.contents,
            self.mock_model.get_preprocessor.return_value,
            inference_request.preprocessing_config,
        )
        self.assertEqual([["original", "preprocessed"]], result)

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_content_preprocessing_image_modality(self, mock_split_preprocess):
        """Ensure that IMAGE modality uses default TextPreprocessingConfig internally.

        Note: InferenceRequest currently only supports TEXT and IMAGE preprocessing configs,
        so we cannot test AUDIO and VIDEO modalities even though the code supports them.
        """
        inference_request = InferenceRequest(
            contents=["http://example.com/image.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)
        mock_split_preprocess.return_value = [[("original", "preprocessed")]]

        result = pipeline._content_preprocessing()

        # Verify split_prefix_preprocess_text was called with default TextPreprocessingConfig
        self.assertEqual(1, mock_split_preprocess.call_count)
        call_args = mock_split_preprocess.call_args
        self.assertEqual(["http://example.com/image.jpg"], call_args[0][0])
        self.assertEqual(self.mock_model.get_preprocessor.return_value, call_args[0][1])
        # Verify the preprocessing config is a default TextPreprocessingConfig (not the original ImagePreprocessingConfig)
        self.assertIsInstance(call_args[0][2], TextPreprocessingConfig)
        self.assertIsNone(call_args[0][2].text_prefix)
        self.assertEqual([[("original", "preprocessed")]], result)

    def test_collect_valid_content_to_encode_with_valid_strings(self):
        """Test collecting valid string content from preprocessed data."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        preprocessed_content = [
            [("original1", "preprocessed1"), ("original2", "preprocessed2")],
            [("original3", "preprocessed3")],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(3, len(result))
        self.assertEqual(["preprocessed1", "preprocessed2", "preprocessed3"], result)

    def test_collect_valid_content_to_encode_with_errors(self):
        """Test that InferenceErrorModel instances are skipped."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        error = InferenceErrorModel(
            status_code=400, error_code="test_error", error_message="Test error message"
        )

        preprocessed_content = [
            [("original1", "preprocessed1")],
            error,
            [("original2", "preprocessed2")],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(2, len(result))
        self.assertEqual(["preprocessed1", "preprocessed2"], result)

    def test_collect_valid_content_to_encode_with_invalid_type(self):
        """Test that invalid content type in tuples raises ValueError."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Invalid type: integer instead of string
        preprocessed_content = [
            [("original1", 12345)],  # type: ignore
        ]

        with self.assertRaises(ValueError) as context:
            pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertIn("Expected", str(context.exception))
        self.assertIn("int", str(context.exception))

    def test_collect_valid_content_to_encode_with_invalid_chunk_type(self):
        """Test that invalid chunk type raises ValueError."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Invalid chunk type: string instead of list or InferenceErrorModel
        preprocessed_content = [
            "invalid_chunk_type",  # type: ignore
        ]

        with self.assertRaises(ValueError) as context:
            pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertIn("Unexpected content type", str(context.exception))

    def test_encode_processed_content_single_batch(self):
        """Test encoding content that fits in a single batch."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        preprocessed_content = [
            [("original1", "preprocessed1")],
            [("original2", "preprocessed2")],
        ]

        # Mock embeddings
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        self.mock_model.encode.return_value = [embedding1, embedding2]

        result = pipeline._encode_processed_content(preprocessed_content)

        self.assertEqual(2, len(result))
        np.testing.assert_array_equal(embedding1, result[0])
        np.testing.assert_array_equal(embedding2, result[1])

        self.mock_model.encode.assert_called_once_with(
            inputs=["preprocessed1", "preprocessed2"],
            modality=Modality.TEXT,
            normalize=True,
        )

    def test_encode_processed_content_multiple_batches(self):
        """Test encoding content that requires multiple batches."""
        # Set small batch size to force multiple batches
        self.mock_model.model_properties.triton_text_encoder_properties.max_batch_size = 2

        # Create config with normalize_embeddings=False
        embedding_config = EmbeddingModelConfig(
            model_name="test-model", normalize_embeddings=False
        )

        inference_request = InferenceRequest(
            contents=["test1", "test2", "test3", "test4", "test5"],
            modality=Modality.TEXT,
            embedding_model_config=embedding_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        preprocessed_content = [
            [("original1", "preprocessed1")],
            [("original2", "preprocessed2")],
            [("original3", "preprocessed3")],
            [("original4", "preprocessed4")],
            [("original5", "preprocessed5")],
        ]

        # Mock embeddings for each batch
        batch1_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        batch2_embeddings = [np.array([0.5, 0.6]), np.array([0.7, 0.8])]
        batch3_embeddings = [np.array([0.9, 1.0])]

        self.mock_model.encode.side_effect = [
            batch1_embeddings,
            batch2_embeddings,
            batch3_embeddings,
        ]

        result = pipeline._encode_processed_content(preprocessed_content)

        self.assertEqual(5, len(result))
        self.assertEqual(3, self.mock_model.encode.call_count)

        # Verify batches
        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(
            ["preprocessed1", "preprocessed2"], call_args_list[0][1]["inputs"]
        )
        self.assertEqual(
            ["preprocessed3", "preprocessed4"], call_args_list[1][1]["inputs"]
        )
        self.assertEqual(["preprocessed5"], call_args_list[2][1]["inputs"])

    def test_encode_processed_content_empty_list(self):
        """Test encoding with empty preprocessed content list returns empty list."""
        # We can't create an InferenceRequest with empty contents due to Pydantic validation,
        # but we can test the _encode_processed_content method directly with empty preprocessed content
        inference_request = InferenceRequest(
            contents=["test"],  # Valid request to pass Pydantic validation
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Test with empty preprocessed content (not empty inference request)
        result = pipeline._encode_processed_content([])

        self.assertEqual([], result)
        self.mock_model.encode.assert_not_called()

    def test_encode_processed_content_mismatch_raises_error(self):
        """Test that mismatch between embeddings and content raises InternalServerError."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        preprocessed_content = [
            [("original1", "preprocessed1")],
            [("original2", "preprocessed2")],
        ]

        # Return wrong number of embeddings
        self.mock_model.encode.return_value = [
            np.array([0.1, 0.2])
        ]  # Only 1 embedding for 2 inputs

        with self.assertRaises(InternalServerError) as context:
            pipeline._encode_processed_content(preprocessed_content)

        self.assertIn("does not match", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_run_pipeline_success(self, mock_split_preprocess):
        """Test the full pipeline execution with successful results."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing
        mock_split_preprocess.return_value = [
            [("test1", "preprocessed1")],
            [("test2", "preprocessed2")],
        ]

        # Mock encoding
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        self.mock_model.encode.return_value = [embedding1, embedding2]

        result = pipeline.run_pipeline()

        # Verify result structure
        self.assertEqual(2, len(result.result))
        self.assertIsInstance(result.result[0], list)
        self.assertIsInstance(result.result[1], list)

        # Verify embeddings in results
        self.assertEqual("test1", result.result[0][0][0])
        np.testing.assert_array_equal(embedding1, result.result[0][0][1])
        self.assertEqual("test2", result.result[1][0][0])
        np.testing.assert_array_equal(embedding2, result.result[1][0][1])

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_run_pipeline_with_errors(self, mock_split_preprocess):
        """Test the full pipeline execution with some errors in preprocessing."""
        inference_request = InferenceRequest(
            contents=["test1", "test2", "test3"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing with an error for the second content
        error = InferenceErrorModel(
            status_code=400,
            error_code="preprocessing_error",
            error_message="Failed to preprocess content",
        )

        mock_split_preprocess.return_value = [
            [("test1", "preprocessed1")],
            error,
            [("test3", "preprocessed3")],
        ]

        # Mock encoding (only 2 embeddings for 2 valid contents)
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        self.mock_model.encode.return_value = [embedding1, embedding2]

        result = pipeline.run_pipeline()

        # Verify result structure
        self.assertEqual(3, len(result.result))

        # First result should be valid
        self.assertIsInstance(result.result[0], list)
        self.assertEqual("test1", result.result[0][0][0])
        np.testing.assert_array_equal(embedding1, result.result[0][0][1])

        # Second result should be an error
        self.assertIsInstance(result.result[1], InferenceErrorModel)
        self.assertEqual("preprocessing_error", result.result[1].error_code)

        # Third result should be valid
        self.assertIsInstance(result.result[2], list)
        self.assertEqual("test3", result.result[2][0][0])
        np.testing.assert_array_equal(embedding2, result.result[2][0][1])

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_run_pipeline_with_chunked_content(self, mock_split_preprocess):
        """Test the pipeline with chunked content (multiple chunks per content)."""
        # Create config with normalize_embeddings=False
        embedding_config = EmbeddingModelConfig(
            model_name="test-model", normalize_embeddings=False
        )

        inference_request = InferenceRequest(
            contents=["long text that gets chunked"],
            modality=Modality.TEXT,
            embedding_model_config=embedding_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = HuggingFaceModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing with multiple chunks
        mock_split_preprocess.return_value = [
            [
                ("long text", "preprocessed_chunk1"),
                ("that gets", "preprocessed_chunk2"),
                ("chunked", "preprocessed_chunk3"),
            ]
        ]

        # Mock encoding
        embedding1 = np.array([0.1, 0.2])
        embedding2 = np.array([0.3, 0.4])
        embedding3 = np.array([0.5, 0.6])
        self.mock_model.encode.return_value = [embedding1, embedding2, embedding3]

        result = pipeline.run_pipeline()

        # Verify result structure
        self.assertEqual(1, len(result.result))
        self.assertEqual(3, len(result.result[0]))

        # Verify each chunk
        self.assertEqual("long text", result.result[0][0][0])
        np.testing.assert_array_equal(embedding1, result.result[0][0][1])
        self.assertEqual("that gets", result.result[0][1][0])
        np.testing.assert_array_equal(embedding2, result.result[0][1][1])
        self.assertEqual("chunked", result.result[0][2][0])
        np.testing.assert_array_equal(embedding3, result.result[0][2][1])


if __name__ == "__main__":
    unittest.main()
