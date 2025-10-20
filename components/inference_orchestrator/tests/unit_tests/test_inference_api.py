import unittest
from unittest.mock import MagicMock, Mock, patch

import msgpack
import numpy as np
from fastapi.testclient import TestClient
from starlette import status
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from inference_orchestrator.main import app
from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    Inference,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.errors import ServiceError


class TestInferenceAPI(unittest.TestCase):
    def setUp(self):
        # Initialize TestClient
        self.client = TestClient(app)

        self.mock_inference = MagicMock(spec=Inference)
        # Patch the _config dependency
        patcher = patch("inference_orchestrator.config._config", autospec=True)
        self.mock_config = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_config.inference = self.mock_inference

    def test_vectorise_success(self):
        self.mock_inference.vectorise.side_effect = [
            InferenceResult(result=[[("chunk", np.array([0.1, 0.2, 0.3]))]])
        ]

        # Prepare a valid InferenceRequest
        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            embedding_model_config=EmbeddingModelConfig(model_name="random"),
            preprocessing_config=TextPreprocessingConfig(),
        )
        packed_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            content=packed_data,
        )

        self.assertEqual(response.status_code, HTTP_200_OK)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("result", unpacked_response)
        self.assertEqual(1, len(unpacked_response["result"]))  # result for one content
        self.assertEqual(1, len(unpacked_response["result"][0]))  # only one chunk
        self.assertEqual(
            2, len(unpacked_response["result"][0][0])
        )  # two elements, chunk key and embeddings
        self.assertEqual(
            "chunk", unpacked_response["result"][0][0][0]
        )  # first element of first chunk
        self.assertTrue(
            np.array_equal([0.1, 0.2, 0.3], unpacked_response["result"][0][0][1])
        )
        self.mock_inference.vectorise.assert_called_once_with(inference_request)

    def test_vectorise_invalid_msgpack_request(self):
        # Send invalid MessagePack data
        invalid_data = b"not a valid msgpack"

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            content=invalid_data,
        )

        self.assertEqual(response.status_code, HTTP_400_BAD_REQUEST)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("Invalid MessagePack format", unpacked_response["detail"])

    def test_vectorise_validation_error(self):
        # Prepare data that fails Pydantic validation (missing required fields)
        invalid_request_data = {"invalid_field": "value"}
        packed_data = msgpack.packb(invalid_request_data, use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            content=packed_data,
        )

        self.assertEqual(response.status_code, HTTP_422_UNPROCESSABLE_CONTENT)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn(
            "4 validation errors for InferenceRequest",
            unpacked_response["detail"],
        )

    def test_vectorise_raise_inference_error(self):
        # Configure the mock to raise an Exception
        self.mock_inference.vectorise.side_effect = ServiceError("Inference failed")

        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            embedding_model_config=EmbeddingModelConfig(model_name="random"),
            preprocessing_config=TextPreprocessingConfig(),
        )
        packed_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            content=packed_data,
        )

        self.assertEqual(response.status_code, HTTP_400_BAD_REQUEST)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn(
            "An error occurred during vectorisation. Inference failed",
            unpacked_response["detail"],
        )

    def test_vectorise_raise_exception_other_than_inference_error(self):
        self.mock_inference.vectorise.side_effect = ValueError("Some internal error")

        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            embedding_model_config=EmbeddingModelConfig(model_name="random"),
            preprocessing_config=TextPreprocessingConfig(),
        )
        packed_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            content=packed_data,
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("Some internal error", unpacked_response["detail"])

    def test_vectorise_unsupported_content_type(self):
        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            embedding_model_config=EmbeddingModelConfig(model_name="random"),
            preprocessing_config=TextPreprocessingConfig(),
        )
        packed_data = msgpack.packb(
            inference_request.model_dump(by_alias=True), use_bin_type=True
        )

        response = self.client.post(
            "/vectorise",
            headers={
                "Content-Type": "application/protobuf",
                "Accept": "application/msgpack",
            },
            content=packed_data,
        )

        self.assertEqual(response.status_code, HTTP_415_UNSUPPORTED_MEDIA_TYPE)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("Unsupported Content-Type", unpacked_response["detail"])

    def test_healthz_happy_pass(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("inference_orchestrator.main.bootstrap_otel")
    def test_lifespan_integration_bootstrap_and_shutdown_otel(
        self, mock_bootstrap_otel
    ):
        mock_otel_shutdown_hook = Mock()
        mock_bootstrap_otel.return_value = mock_otel_shutdown_hook

        # Use FastAPI TestClient to simulate making a request to the app
        with TestClient(app) as _:
            # Ensure the shutdown hook was called and Zookeeper stop method was triggered
            mock_bootstrap_otel.assert_called_once_with(
                app, service_name="marqo-inference"
            )

        mock_otel_shutdown_hook.assert_called_once()

    @patch("inference_orchestrator.main.get_config")
    def test_lifespan_integration_shutdown_triton_client(self, mock_get_config):
        mock_triton_client = Mock()
        mock_config = Mock()
        mock_config.triton_client = mock_triton_client
        mock_get_config.return_value = mock_config

        # Use FastAPI TestClient to simulate making a request to the app
        with TestClient(app) as _:
            # Ensure get_config was called
            mock_get_config.assert_called()

        # Ensure the triton client close method was called during shutdown
        mock_triton_client.close.assert_called_once()
