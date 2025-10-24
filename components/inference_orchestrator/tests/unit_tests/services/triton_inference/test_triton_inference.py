import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.errors import InternalServerError
from inference_orchestrator.services.triton_inference.embedding_models import (
    HuggingFaceModel,
    OpenCLIPModel,
    RandomModel,
)
from inference_orchestrator.services.triton_inference.triton_inference import (
    TritonInference,
)


class TestTritonInference(unittest.TestCase):
    """Test suite for TritonInference class.

    These tests verify the TritonInference.vectorise() method correctly:
    - Loads the appropriate model based on the request
    - Routes to the correct inference pipeline based on model type
    - Returns InferenceResult from the pipeline
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model_management_client = MagicMock()
        self.mock_triton_client = MagicMock()
        self.triton_inference = TritonInference(
            model_management_client=self.mock_model_management_client,
            triton_client=self.mock_triton_client,
        )

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.OpenCLIPModelInferencePipeline"
    )
    def test_vectorise_with_openclip_model(self, mock_pipeline_class, mock_load_model):
        """Test vectorise() with OpenCLIP model routes to OpenCLIPModelInferencePipeline."""
        # Set up mocks
        mock_model = MagicMock(spec=OpenCLIPModel)
        mock_load_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        expected_result = InferenceResult(
            result=[[("chunk1", np.array([1.0, 2.0, 3.0]))]]
        )
        mock_pipeline_instance.run_pipeline.return_value = expected_result
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Create request
        request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test content"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="test-openclip-model", normalizeEmbeddings=True
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Execute
        result = self.triton_inference.vectorise(request)

        # Verify load_model was called correctly
        mock_load_model.assert_called_once_with(
            model_name="test-openclip-model",
            model_properties=None,
            triton_client=self.mock_triton_client,
            model_management_client=self.mock_model_management_client,
        )

        # Verify pipeline was instantiated and run
        mock_pipeline_class.assert_called_once_with(mock_model, request)
        mock_pipeline_instance.run_pipeline.assert_called_once()

        # Verify result
        self.assertEqual(expected_result, result)

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.HuggingFaceModelInferencePipeline"
    )
    def test_vectorise_with_huggingface_model(
        self, mock_pipeline_class, mock_load_model
    ):
        """Test vectorise() with HuggingFace model routes to HuggingFaceModelInferencePipeline."""
        # Set up mocks
        mock_model = MagicMock(spec=HuggingFaceModel)
        mock_load_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        expected_result = InferenceResult(
            result=[[("chunk1", np.array([0.5, 0.5, 0.5]))]]
        )
        mock_pipeline_instance.run_pipeline.return_value = expected_result
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Create request
        request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["huggingface content"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="test-hf-model", normalizeEmbeddings=False
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Execute
        result = self.triton_inference.vectorise(request)

        # Verify load_model was called correctly
        mock_load_model.assert_called_once_with(
            model_name="test-hf-model",
            model_properties=None,
            triton_client=self.mock_triton_client,
            model_management_client=self.mock_model_management_client,
        )

        # Verify pipeline was instantiated and run
        mock_pipeline_class.assert_called_once_with(mock_model, request)
        mock_pipeline_instance.run_pipeline.assert_called_once()

        # Verify result
        self.assertEqual(expected_result, result)

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.RandomModelInferencePipeline"
    )
    def test_vectorise_with_random_model(self, mock_pipeline_class, mock_load_model):
        """Test vectorise() with Random model routes to RandomModelInferencePipeline."""
        # Set up mocks
        mock_model = MagicMock(spec=RandomModel)
        mock_load_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        expected_result = InferenceResult(result=[[("chunk1", np.random.rand(128))]])
        mock_pipeline_instance.run_pipeline.return_value = expected_result
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Create request
        request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["random model content"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="random", normalizeEmbeddings=True
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Execute
        result = self.triton_inference.vectorise(request)

        # Verify load_model was called correctly
        mock_load_model.assert_called_once_with(
            model_name="random",
            model_properties=None,
            triton_client=self.mock_triton_client,
            model_management_client=self.mock_model_management_client,
        )

        # Verify pipeline was instantiated and run
        mock_pipeline_class.assert_called_once_with(mock_model, request)
        mock_pipeline_instance.run_pipeline.assert_called_once()

        # Verify result
        self.assertEqual(expected_result, result)

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    def test_vectorise_with_unsupported_model_raises_error(self, mock_load_model):
        """Test vectorise() raises InternalServerError for unsupported model types."""

        # Create a mock model that's not one of the supported types
        class UnsupportedModel:
            pass

        mock_model = UnsupportedModel()
        mock_load_model.return_value = mock_model

        # Create request
        request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test content"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="unsupported-model", normalizeEmbeddings=True
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Execute and verify
        with self.assertRaises(InternalServerError) as context:
            self.triton_inference.vectorise(request)

        self.assertIn("Model not supported", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.OpenCLIPModelInferencePipeline"
    )
    def test_vectorise_with_model_properties(
        self, mock_pipeline_class, mock_load_model
    ):
        """Test vectorise() passes model_properties to load_model correctly."""
        # Set up mocks
        mock_model = MagicMock(spec=OpenCLIPModel)
        mock_load_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        expected_result = InferenceResult(result=[[]])
        mock_pipeline_instance.run_pipeline.return_value = expected_result
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Create request with model properties
        model_properties = {
            "type": "open_clip",
            "dimensions": 512,
            "imagePreprocessor": "SigLIP",
        }
        request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=["https://example.com/image.jpg"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="test-model",
                modelProperties=model_properties,
                normalizeEmbeddings=True,
            ),
            preprocessingConfig=ImagePreprocessingConfig(modality=Modality.IMAGE),
        )

        # Execute
        result = self.triton_inference.vectorise(request)

        # Verify load_model was called with model_properties
        mock_load_model.assert_called_once_with(
            model_name="test-model",
            model_properties=model_properties,
            triton_client=self.mock_triton_client,
            model_management_client=self.mock_model_management_client,
        )

        # Verify result
        self.assertEqual(expected_result, result)

    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.load_model"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.triton_inference.HuggingFaceModelInferencePipeline"
    )
    def test_vectorise_uses_correct_clients(self, mock_pipeline_class, mock_load_model):
        """Test that vectorise() uses the clients passed during initialization."""
        # Set up mocks
        mock_model = MagicMock(spec=HuggingFaceModel)
        mock_load_model.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        expected_result = InferenceResult(result=[[]])
        mock_pipeline_instance.run_pipeline.return_value = expected_result
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Create request
        request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["test"],
            embeddingModelConfig=EmbeddingModelConfig(
                modelName="test-model", normalizeEmbeddings=True
            ),
            preprocessingConfig=TextPreprocessingConfig(modality=Modality.TEXT),
        )

        # Execute
        self.triton_inference.vectorise(request)

        # Verify the clients from __init__ were used
        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        self.assertIs(call_kwargs["triton_client"], self.mock_triton_client)
        self.assertIs(
            call_kwargs["model_management_client"], self.mock_model_management_client
        )


if __name__ == "__main__":
    unittest.main()
