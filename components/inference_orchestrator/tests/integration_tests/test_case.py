from enum import StrEnum
from typing import List, Optional, Union
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy import ndarray

from inference_orchestrator.api.telemetry import RequestMetricsStore
from inference_orchestrator.config import get_config
from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.triton_inference.embedding_models.marqo_model_registry import (
    get_model_properties,
)
from inference_orchestrator.services.triton_inference.model_manager import model_manager
from inference_orchestrator.services.triton_inference.triton_inference import (
    TritonInference,
)


class TestImageUrls(StrEnum):
    __test__ = False  # Prevent pytest from collecting this class as a test
    IMAGE0 = "https://marqo-assets.s3.amazonaws.com/tests/images/image0.jpg"
    IMAGE1 = "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg"
    IMAGE2 = "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg"
    IMAGE3 = "https://marqo-assets.s3.amazonaws.com/tests/images/image3.jpg"
    IMAGE4 = "https://marqo-assets.s3.amazonaws.com/tests/images/image4.jpg"
    HIPPO_REALISTIC = "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic_small.png"
    HIPPO_REALISTIC_LARGE = "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
    HIPPO_STATUE = "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue_small.png"


class InferenceTestCase(TestCase):
    @classmethod
    def configure_request_metrics(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics."""
        cls.mock_request = Mock()
        cls.patcher = patch(
            "inference_orchestrator.api.telemetry.RequestMetricsStore._get_request"
        )
        cls.mock_get_request = cls.patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

    @classmethod
    def setUpClass(cls) -> None:
        cls.configure_request_metrics()
        cls.config = get_config()
        cls.inference = cls.config.inference

    def validate_norm(
        self, embedding: ndarray, epsilon: float = 1e-6, normalize: bool = True
    ):
        if normalize:
            return self.assertTrue(
                abs(np.linalg.norm(embedding) - 1) < epsilon, np.linalg.norm(embedding)
            )
        else:
            return self.assertTrue(
                abs(np.linalg.norm(embedding) - 1) > epsilon, np.linalg.norm(embedding)
            )

    def get_model_properties_from_registry(self, model_name: str) -> dict:
        return get_model_properties(model_name)

    def encode_content_helper(
        self,
        content: list[str],
        model_name: str = "",
        model_properties: Optional[dict] = None,
        modality: Union[Modality, str] = Modality.TEXT,
        normalize_embeddings: bool = True,
        media_download_headers: Optional[dict] = None,
    ) -> List[ndarray]:
        """
        A helper function to encode the content of a document.

        Args:
            content: A list of strings to be encoded, can be text or media URLs.
            model_name: The name of the model to be used for encoding.
            model_properties: A dictionary containing the properties of the model to be used for encoding. If this is
                None, model_properties will be retrieved from the model registry and model_name must be a valid model
                in the registry. If this is not None, model_name is ignored and we will use the model properties
                to load the model.
            modality: The modality of the content to be encoded. Default is TEXT.
            normalize_embeddings: Whether to normalize the embeddings. Default is True.
            media_download_headers: The headers to be used for downloading media. Default is None.
        Returns:
            A list of numpy arrays, each representing the encoded content, with dimensions (Dim, ).
        """
        if not model_name and not model_properties:
            raise ValueError("Either model_name or model_properties must be provided.")

        if not model_properties:
            model_properties = self.get_model_properties_from_registry(model_name)

        if modality == Modality.TEXT:
            preprocessing_config = TextPreprocessingConfig(should_chunk=False)
        elif modality == Modality.IMAGE:
            preprocessing_config = ImagePreprocessingConfig(
                download_thread_count=1, download_header=media_download_headers
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        inference_request = InferenceRequest(
            modality=modality,
            contents=content,
            embedding_model_config=EmbeddingModelConfig(
                model_name=model_name,
                model_properties=model_properties,
                normalize_embeddings=normalize_embeddings,
            ),
            use_inference_cache=False,
            return_individual_error=True,
            preprocessing_config=preprocessing_config,
        )

        results: InferenceResult = TritonInference(
            model_management_client=self.config.model_management_client,
            triton_client=self.config.triton_client,
        ).vectorise(inference_request)
        embeddings = [result[0][1] for result in results.result]
        return embeddings

    def calculate_embeddings_difference(
        self, embedding_1: ndarray, embedding_2: ndarray
    ):
        """
        Calculate the difference between two embeddings.

        We use the mean absolute error (MAE) to calculate the difference between the two embeddings after normalizing
        them.
        """

        if embedding_1.shape != embedding_2.shape:
            raise ValueError("The two embeddings must have the same shape.")

        embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
        embedding_2 = embedding_2 / np.linalg.norm(embedding_2)

        return np.mean(np.abs(embedding_1 - embedding_2))

    @staticmethod
    def eject_all_models():
        """Eject all models from Triton to ensure a clean state for tests that need to load models."""
        for model_name in [
            model["modelName"] for model in model_manager.get_loaded_models()["models"]
        ]:
            model_manager.eject_model(model_name)
