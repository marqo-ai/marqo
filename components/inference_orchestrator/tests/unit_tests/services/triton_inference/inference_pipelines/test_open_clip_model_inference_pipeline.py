import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline,
)


class TestOpenCLIPModelInferencePipeline(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.get_preprocessor.return_value = MagicMock()

        # Mock model properties for batching - OpenCLIP has separate batch sizes for text and image
        self.mock_model.model_properties.triton_text_encoder_properties.max_batch_size = 32
        self.mock_model.model_properties.triton_image_encoder_properties.max_batch_size = 16

        # Create a real EmbeddingModelConfig for Pydantic validation
        self.embedding_model_config = EmbeddingModelConfig(
            model_name="test-open-clip-model", normalize_embeddings=True
        )

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_content_preprocessing_text_modality(self, mock_split_preprocess):
        """Ensure that the content preprocessing is done correctly for text modalities."""
        inference_request = InferenceRequest(
            contents=["this is a test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)
        mock_split_preprocess.return_value = [["original", "preprocessed"]]

        result = pipeline._content_preprocessing()

        mock_split_preprocess.assert_called_once_with(
            inference_request.contents,
            self.mock_model.get_preprocessor.return_value,
            inference_request.preprocessing_config,
        )
        self.assertEqual([["original", "preprocessed"]], result)

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.download_and_preprocess_media"
    )
    def test_content_preprocessing_image_modality(self, mock_download_preprocess):
        """Ensure that IMAGE modality uses download_and_preprocess_media."""
        inference_request = InferenceRequest(
            contents=["http://example.com/image.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        expected_tensor = torch.tensor([1.0, 2.0])
        mock_download_preprocess.return_value = [[("original", expected_tensor)]]

        result = pipeline._content_preprocessing()

        mock_download_preprocess.assert_called_once_with(
            inference_request.contents,
            self.mock_model.get_preprocessor.return_value,
            inference_request.preprocessing_config,
            inference_request.return_individual_error,
        )

        # Verify structure and content
        self.assertEqual(1, len(result))
        self.assertEqual(1, len(result[0]))
        self.assertEqual("original", result[0][0][0])
        self.assertTrue(torch.equal(expected_tensor, result[0][0][1]))

    def test_collect_valid_content_to_encode_with_tensors(self):
        """Test collecting valid tensor content from preprocessed data."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        tensor3 = torch.tensor([5.0, 6.0])

        preprocessed_content = [
            [("original1", tensor1), ("original2", tensor2)],
            [("original3", tensor3)],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(3, len(result))
        self.assertTrue(torch.equal(tensor1, result[0]))
        self.assertTrue(torch.equal(tensor2, result[1]))
        self.assertTrue(torch.equal(tensor3, result[2]))

    def test_collect_valid_content_to_encode_with_strings(self):
        """Test collecting valid string content (for text modality) from preprocessed data."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

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
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        error = InferenceErrorModel(
            status_code=400, error_code="test_error", error_message="Test error message"
        )

        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])

        preprocessed_content = [
            [("original1", tensor1)],
            error,
            [("original2", tensor2)],
        ]

        result = pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertEqual(2, len(result))
        self.assertTrue(torch.equal(tensor1, result[0]))
        self.assertTrue(torch.equal(tensor2, result[1]))

    def test_collect_valid_content_to_encode_with_invalid_type(self):
        """Test that invalid content type in tuples raises ValueError."""
        inference_request = InferenceRequest(
            contents=["test"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Invalid type: integer instead of tensor or string
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
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Invalid chunk type: string instead of list or InferenceErrorModel
        preprocessed_content = [
            "invalid_chunk_type",  # type: ignore
        ]

        with self.assertRaises(ValueError) as context:
            pipeline._collect_valid_content_to_encode(preprocessed_content)

        self.assertIn("Unexpected content type", str(context.exception))

    def test_encode_processed_content_single_batch_text_modality(self):
        """Test encoding text content that fits in a single batch."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])

        preprocessed_content = [
            [("original1", tensor1)],
            [("original2", tensor2)],
        ]

        # Mock embeddings
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        self.mock_model.encode.return_value = [embedding1, embedding2]

        result = pipeline._encode_processed_content(preprocessed_content)

        self.assertEqual(2, len(result))
        np.testing.assert_array_equal(embedding1, result[0])
        np.testing.assert_array_equal(embedding2, result[1])

        # Verify text encoder batch size was used
        self.mock_model.encode.assert_called_once_with(
            inputs=[tensor1, tensor2],
            modality=Modality.TEXT,
            normalize=True,
        )

    def test_encode_processed_content_single_batch_image_modality(self):
        """Test encoding image content that fits in a single batch."""
        inference_request = InferenceRequest(
            contents=["http://example.com/image.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensor1 = torch.tensor([1.0, 2.0])

        preprocessed_content = [
            [("original1", tensor1)],
        ]

        # Mock embeddings
        embedding1 = np.array([0.1, 0.2, 0.3])
        self.mock_model.encode.return_value = [embedding1]

        result = pipeline._encode_processed_content(preprocessed_content)

        self.assertEqual(1, len(result))
        np.testing.assert_array_equal(embedding1, result[0])

        # Verify image encoder batch size was used
        self.mock_model.encode.assert_called_once_with(
            inputs=[tensor1],
            modality=Modality.IMAGE,
            normalize=True,
        )

    def test_encode_processed_content_multiple_batches_text_modality(self):
        """Test encoding text content that requires multiple batches."""
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

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensors = [torch.tensor([float(i)]) for i in range(5)]

        preprocessed_content = [
            [("original1", tensors[0])],
            [("original2", tensors[1])],
            [("original3", tensors[2])],
            [("original4", tensors[3])],
            [("original5", tensors[4])],
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
        self.assertEqual(2, len(call_args_list[0][1]["inputs"]))
        self.assertEqual(2, len(call_args_list[1][1]["inputs"]))
        self.assertEqual(1, len(call_args_list[2][1]["inputs"]))

    def test_encode_processed_content_multiple_batches_image_modality(self):
        """Test encoding image content that requires multiple batches."""
        # Set small batch size to force multiple batches
        self.mock_model.model_properties.triton_image_encoder_properties.max_batch_size = 3

        inference_request = InferenceRequest(
            contents=[
                "img1.jpg",
                "img2.jpg",
                "img3.jpg",
                "img4.jpg",
                "img5.jpg",
                "img6.jpg",
                "img7.jpg",
            ],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensors = [torch.tensor([float(i)]) for i in range(7)]

        preprocessed_content = [[("original" + str(i), tensors[i])] for i in range(7)]

        # Mock embeddings for each batch (3, 3, 1)
        batch1_embeddings = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        batch2_embeddings = [np.array([0.4]), np.array([0.5]), np.array([0.6])]
        batch3_embeddings = [np.array([0.7])]

        self.mock_model.encode.side_effect = [
            batch1_embeddings,
            batch2_embeddings,
            batch3_embeddings,
        ]

        result = pipeline._encode_processed_content(preprocessed_content)

        self.assertEqual(7, len(result))
        self.assertEqual(3, self.mock_model.encode.call_count)

        # Verify batch sizes and modality
        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(3, len(call_args_list[0][1]["inputs"]))
        self.assertEqual(Modality.IMAGE, call_args_list[0][1]["modality"])
        self.assertEqual(3, len(call_args_list[1][1]["inputs"]))
        self.assertEqual(1, len(call_args_list[2][1]["inputs"]))

    def test_encode_processed_content_empty_list(self):
        """Test encoding with empty preprocessed content list returns empty list."""
        inference_request = InferenceRequest(
            contents=["test"],  # Valid request to pass Pydantic validation
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Test with empty preprocessed content (not empty inference request)
        result = pipeline._encode_processed_content([])

        self.assertEqual([], result)
        self.mock_model.encode.assert_not_called()

    def test_encode_processed_content_mismatch_raises_error(self):
        """Test that mismatch between embeddings and content raises ValueError."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        tensor1 = torch.tensor([1.0])
        tensor2 = torch.tensor([2.0])

        preprocessed_content = [
            [("original1", tensor1)],
            [("original2", tensor2)],
        ]

        # Return wrong number of embeddings
        self.mock_model.encode.return_value = [
            np.array([0.1, 0.2])
        ]  # Only 1 embedding for 2 inputs

        with self.assertRaises(ValueError) as context:
            pipeline._encode_processed_content(preprocessed_content)

        self.assertIn("does not match", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_run_pipeline_success_text_modality(self, mock_split_preprocess):
        """Test the full pipeline execution with successful results for text modality."""
        inference_request = InferenceRequest(
            contents=["test1", "test2"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing
        tensor1 = torch.tensor([1.0])
        tensor2 = torch.tensor([2.0])
        mock_split_preprocess.return_value = [
            [("test1", tensor1)],
            [("test2", tensor2)],
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
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.download_and_preprocess_media"
    )
    def test_run_pipeline_success_image_modality(self, mock_download_preprocess):
        """Test the full pipeline execution with successful results for image modality."""
        inference_request = InferenceRequest(
            contents=["http://example.com/img1.jpg", "http://example.com/img2.jpg"],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        mock_download_preprocess.return_value = [
            [("http://example.com/img1.jpg", tensor1)],
            [("http://example.com/img2.jpg", tensor2)],
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
        self.assertEqual("http://example.com/img1.jpg", result.result[0][0][0])
        np.testing.assert_array_equal(embedding1, result.result[0][0][1])
        self.assertEqual("http://example.com/img2.jpg", result.result[1][0][0])
        np.testing.assert_array_equal(embedding2, result.result[1][0][1])

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.split_prefix_preprocess_text"
    )
    def test_run_pipeline_with_errors(self, mock_split_preprocess):
        """Test the full pipeline execution with some errors in preprocessing."""
        inference_request = InferenceRequest(
            contents=["test1", "test2", "test3"],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing with an error for the second content
        error = InferenceErrorModel(
            status_code=400,
            error_code="preprocessing_error",
            error_message="Failed to preprocess content",
        )

        tensor1 = torch.tensor([1.0])
        tensor3 = torch.tensor([3.0])

        mock_split_preprocess.return_value = [
            [("test1", tensor1)],
            error,
            [("test3", tensor3)],
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
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.split_prefix_preprocess_text"
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

        pipeline = OpenCLIPModelInferencePipeline(self.mock_model, inference_request)

        # Mock preprocessing with multiple chunks
        tensor1 = torch.tensor([1.0])
        tensor2 = torch.tensor([2.0])
        tensor3 = torch.tensor([3.0])

        mock_split_preprocess.return_value = [
            [
                ("long text", tensor1),
                ("that gets", tensor2),
                ("chunked", tensor3),
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

    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.split_prefix_preprocess_text"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline.download_and_preprocess_media"
    )
    def test_different_batch_sizes_for_text_and_image(
        self, mock_download_preprocess, mock_split_preprocess
    ):
        """Test that text and image modalities use different max batch sizes and batch content accordingly."""
        # Set specific batch sizes
        self.mock_model.model_properties.triton_text_encoder_properties.max_batch_size = 5
        self.mock_model.model_properties.triton_image_encoder_properties.max_batch_size = 3

        # Test TEXT modality with 12 items (should create 3 batches: 5, 5, 2)
        text_request = InferenceRequest(
            contents=["text" + str(i) for i in range(12)],
            modality=Modality.TEXT,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        text_pipeline = OpenCLIPModelInferencePipeline(self.mock_model, text_request)

        # Mock preprocessing to return tensors
        text_tensors = [torch.tensor([float(i)]) for i in range(12)]
        mock_split_preprocess.return_value = [
            [("text" + str(i), text_tensors[i])] for i in range(12)
        ]

        # Mock embeddings for text batches (3 batches: 5, 5, 2)
        text_batch1 = [np.array([0.1 * i]) for i in range(5)]
        text_batch2 = [np.array([0.2 * i]) for i in range(5)]
        text_batch3 = [np.array([0.3 * i]) for i in range(2)]
        self.mock_model.encode.side_effect = [text_batch1, text_batch2, text_batch3]

        # Run the pipeline - this should trigger preprocessing and encoding
        result = text_pipeline.run_pipeline()

        # Verify 3 batches were created for text
        self.assertEqual(3, self.mock_model.encode.call_count)
        self.assertEqual(12, len(result.result))

        # Verify batch sizes for text by inspecting encode call arguments
        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(5, len(call_args_list[0][1]["inputs"]))  # First batch: 5
        self.assertEqual(Modality.TEXT, call_args_list[0][1]["modality"])
        self.assertEqual(5, len(call_args_list[1][1]["inputs"]))  # Second batch: 5
        self.assertEqual(Modality.TEXT, call_args_list[1][1]["modality"])
        self.assertEqual(2, len(call_args_list[2][1]["inputs"]))  # Third batch: 2
        self.assertEqual(Modality.TEXT, call_args_list[2][1]["modality"])

        # Reset mocks for image test
        self.mock_model.encode.reset_mock()
        mock_split_preprocess.reset_mock()
        mock_download_preprocess.reset_mock()

        # Test IMAGE modality with 10 items (should create 4 batches: 3, 3, 3, 1)
        image_request = InferenceRequest(
            contents=["img" + str(i) + ".jpg" for i in range(10)],
            modality=Modality.IMAGE,
            embedding_model_config=self.embedding_model_config,
            preprocessing_config=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        image_pipeline = OpenCLIPModelInferencePipeline(self.mock_model, image_request)

        # Mock preprocessing to return tensors
        image_tensors = [torch.tensor([float(i), float(i + 1)]) for i in range(10)]
        mock_download_preprocess.return_value = [
            [("img" + str(i), image_tensors[i])] for i in range(10)
        ]

        # Mock embeddings for image batches (4 batches: 3, 3, 3, 1)
        image_batch1 = [np.array([0.1 * i]) for i in range(3)]
        image_batch2 = [np.array([0.2 * i]) for i in range(3)]
        image_batch3 = [np.array([0.3 * i]) for i in range(3)]
        image_batch4 = [np.array([0.4])]
        self.mock_model.encode.side_effect = [
            image_batch1,
            image_batch2,
            image_batch3,
            image_batch4,
        ]

        # Run the pipeline - this should trigger preprocessing and encoding
        result = image_pipeline.run_pipeline()

        # Verify 4 batches were created for images
        self.assertEqual(4, self.mock_model.encode.call_count)
        self.assertEqual(10, len(result.result))

        # Verify batch sizes for images by inspecting encode call arguments
        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(3, len(call_args_list[0][1]["inputs"]))  # First batch: 3
        self.assertEqual(Modality.IMAGE, call_args_list[0][1]["modality"])
        self.assertEqual(3, len(call_args_list[1][1]["inputs"]))  # Second batch: 3
        self.assertEqual(Modality.IMAGE, call_args_list[1][1]["modality"])
        self.assertEqual(3, len(call_args_list[2][1]["inputs"]))  # Third batch: 3
        self.assertEqual(Modality.IMAGE, call_args_list[2][1]["modality"])
        self.assertEqual(1, len(call_args_list[3][1]["inputs"]))  # Fourth batch: 1
        self.assertEqual(Modality.IMAGE, call_args_list[3][1]["modality"])


if __name__ == "__main__":
    unittest.main()
