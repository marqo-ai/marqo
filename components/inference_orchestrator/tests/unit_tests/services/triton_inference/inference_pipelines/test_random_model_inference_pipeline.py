import unittest
from unittest.mock import patch

import numpy as np

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.triton_inference.embedding_models.random.random_model import (
    RandomModel,
)
from inference_orchestrator.services.triton_inference.inference_pipelines.random_model_inference_pipeline import (
    RandomModelInferencePipeline,
)


class TestRandomModelInferencePipeline(unittest.TestCase):
    """Tests for RandomModelInferencePipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a real RandomModel instance for testing
        model_properties = {
            "type": "random",
            "name": "test-random-model",
            "dimensions": 128,
        }
        self.model = RandomModel(model_properties)

        self.text_config = EmbeddingModelConfig(
            model_name="test-random-model", normalize_embeddings=True
        )

        self.image_config = EmbeddingModelConfig(
            model_name="test-random-model", normalize_embeddings=True
        )

    def test_init(self):
        """Test RandomModelInferencePipeline initialization."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        self.assertEqual(self.model, pipeline.model)
        self.assertEqual(request, pipeline.inference_request)
        self.assertEqual(128, pipeline.MAX_BATCH_SIZE)

    def test_content_preprocessing_text_modality(self):
        """Test content preprocessing with TEXT modality."""
        request = InferenceRequest(
            contents=["test text 1", "test text 2"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        with patch(
            "inference_orchestrator.services.triton_inference.inference_pipelines.random_model_inference_pipeline.split_prefix_preprocess_text"
        ) as mock_split:
            expected_result = [
                [("original1", "preprocessed1")],
                [("original2", "preprocessed2")],
            ]
            mock_split.return_value = expected_result

            result = pipeline._content_preprocessing()

            mock_split.assert_called_once_with(
                ["test text 1", "test text 2"],
                self.model.get_preprocessor(),
                request.preprocessing_config,
            )
            self.assertEqual(expected_result, result)

    def test_content_preprocessing_image_modality(self):
        """Test content preprocessing with IMAGE modality."""
        request = InferenceRequest(
            contents=["image1.jpg", "image2.jpg", "image3.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.image_config,
            preprocessing_config=ImagePreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )
        result = pipeline._content_preprocessing()

        expected = [
            [("image1.jpg", "image1.jpg")],
            [("image2.jpg", "image2.jpg")],
            [("image3.jpg", "image3.jpg")],
        ]
        self.assertEqual(expected, result)

    def test_collect_valid_content_to_encode_with_valid_content(self):
        """Test collecting valid content to encode."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            [("original1", "content1"), ("original2", "content2")],
            [("original3", "content3")],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed)

        self.assertEqual(["content1", "content2", "content3"], result)

    def test_collect_valid_content_to_encode_with_inference_errors(self):
        """Test that InferenceErrorModel items are skipped."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            [("original1", "content1")],
            InferenceErrorModel(
                status_code=500,
                error_code="TestError",
                error_message="Test error message",
            ),
            [("original2", "content2")],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed)

        self.assertEqual(["content1", "content2"], result)

    def test_collect_valid_content_to_encode_with_invalid_content_type_raises_error(
        self,
    ):
        """Test that invalid content type in tuple raises ValueError."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            [("original", 123)],  # Invalid: should be string
        ]

        with self.assertRaises(ValueError) as context:
            pipeline._collect_valid_content_to_encode(preprocessed)
        self.assertIn("Expected", str(context.exception))

    def test_collect_valid_content_to_encode_with_unexpected_chunk_type_raises_error(
        self,
    ):
        """Test that unexpected chunk type raises ValueError."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            "invalid_chunk_type",  # Should be list or InferenceErrorModel
        ]

        with self.assertRaises(ValueError) as context:
            pipeline._collect_valid_content_to_encode(preprocessed)
        self.assertIn("Unexpected content type", str(context.exception))

    def test_encode_processed_content_all_errors(self):
        """Test encoding with all errors returns empty list."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        # When all content is errors, there's nothing to encode
        preprocessed = [
            InferenceErrorModel(
                status_code=500, error_code="Error", error_message="Error message"
            )
        ]

        result = pipeline._encode_processed_content(preprocessed)

        self.assertEqual([], result)

    def test_encode_processed_content_single_batch(self):
        """Test encoding content within single batch size."""
        request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            [("original1", "content1")],
            [("original2", "content2")],
        ]

        result = pipeline._encode_processed_content(preprocessed)

        self.assertEqual(2, len(result))
        for embedding in result:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual((128,), embedding.shape)

    def test_encode_processed_content_multiple_batches(self):
        """Test encoding content split across multiple batches."""
        # Create request with more items than MAX_BATCH_SIZE
        num_items = 150  # More than MAX_BATCH_SIZE (128)
        contents = [f"content{i}" for i in range(num_items)]

        request = InferenceRequest(
            contents=contents,
            modality=Modality.TEXT,
            embedding_model_config=EmbeddingModelConfig(
                model_name="test-model", normalize_embeddings=False
            ),
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        # Create preprocessed content
        preprocessed = [[("original", f"content{i}")] for i in range(num_items)]

        result = pipeline._encode_processed_content(preprocessed)

        # Should have 150 embeddings
        self.assertEqual(150, len(result))

        # All should be numpy arrays with correct shape
        for embedding in result:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual((128,), embedding.shape)

    def test_encode_processed_content_mismatched_length_raises_error(self):
        """Test that mismatched embedding count raises ValueError."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        preprocessed = [
            [("original1", "content1")],
            [("original2", "content2")],
        ]

        # Mock the model.encode to return wrong number of embeddings
        with patch.object(self.model, "encode") as mock_encode:
            mock_encode.return_value = [np.array([1.0, 2.0])]  # Only 1 instead of 2

            with self.assertRaises(ValueError) as context:
                pipeline._encode_processed_content(preprocessed)
            self.assertIn("does not match", str(context.exception))

    def test_run_pipeline_complete_flow(self):
        """Test the complete run_pipeline method."""
        request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        with patch(
            "inference_orchestrator.services.triton_inference.inference_pipelines.random_model_inference_pipeline.split_prefix_preprocess_text"
        ) as mock_split:
            # Setup mock to return preprocessed content
            mock_split.return_value = [[("test1", "test1")], [("test2", "test2")]]

            result = pipeline.run_pipeline()

            # Verify result type - it returns the formatted result from format_results
            # which can be various types depending on AbstractInferencePipeline implementation
            self.assertIsNotNone(result)

    def test_run_pipeline_with_errors(self):
        """Test run_pipeline handling InferenceErrorModel."""
        request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        with patch(
            "inference_orchestrator.services.triton_inference.inference_pipelines.random_model_inference_pipeline.split_prefix_preprocess_text"
        ) as mock_split:
            # Setup mock to return one success and one error
            mock_split.return_value = [
                [("test1", "test1")],
                InferenceErrorModel(
                    status_code=400,
                    error_code="TestError",
                    error_message="Test error message",
                ),
            ]

            result = pipeline.run_pipeline()

            # Just verify the pipeline completes without crashing
            self.assertIsNotNone(result)

    def test_max_batch_size_constant(self):
        """Test that MAX_BATCH_SIZE is set correctly."""
        request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.text_config,
            preprocessing_config=TextPreprocessingConfig(),
        )

        pipeline = RandomModelInferencePipeline(
            model=self.model, inference_request=request
        )

        self.assertEqual(128, pipeline.MAX_BATCH_SIZE)
        self.assertEqual(128, RandomModelInferencePipeline.MAX_BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
