from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np

from marqo.core.inference.api import *
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.load_model import MODEL_PROPERTIES
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search.telemetry import RequestMetricsStore


class InferenceTestCase(TestCase):

    @classmethod
    def configure_request_metrics(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        """
        cls.mock_request = Mock()
        cls.patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

    @classmethod
    def setUpClass(cls) -> None:
        cls.configure_request_metrics()

    def validate_norm(self, embedding: ndarray, epsilon: float = 1e-6, normalize: bool = True):
        if normalize:
            return self.assertTrue(abs(np.linalg.norm(embedding) - 1) < epsilon, np.linalg.norm(embedding))
        else:
            return self.assertTrue(abs(np.linalg.norm(embedding) - 1) > epsilon, np.linalg.norm(embedding))

    def get_model_properties_from_registry(self, model_name: str) -> dict:
        return MODEL_PROPERTIES["models"][model_name]

    def encode_content_helper(
            self, content: list[str], model_name: str="", model_properties: Optional[dict] = None,
            modality: Union[Modality, str] = Modality.TEXT, device: Optional[str] = "cpu",
            normalize_embeddings: bool = True, media_download_headers: Optional[dict] = None
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
            device: The device to be used for encoding. Default is "cpu".
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
                download_thread_count=1,
                download_header=media_download_headers
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        inference_request = InferenceRequest(
            modality=modality,
            contents=content,
            device=device,
            model_config=ModelConfig(
                model_name=model_name,
                model_properties=model_properties,
                normalize_embeddings=normalize_embeddings
            ),
            use_inference_cache=False,
            return_individual_error=True,
            preprocessing_config=preprocessing_config
        )

        results: InferenceResult = NativeInferenceLocal(DeviceManager()).vectorise(inference_request)
        embeddings = [result[0][1] for result in results.result]
        return embeddings

    def calculate_embeddings_difference(self, embedding_1: ndarray, embedding_2: ndarray):
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