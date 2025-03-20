import unittest
from unittest.mock import patch, MagicMock

import msgpack
import numpy as np
from fastapi.testclient import TestClient
from starlette import status
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
)

from marqo.core.exceptions import CudaDeviceNotAvailableError, CudaOutOfMemoryError
from marqo.core.inference.api import InferenceRequest, Modality, ModelConfig, TextPreprocessingConfig, Inference, \
    InferenceResult, InferenceError
from marqo.inference.native_inference.remote.server.inference_api import app


class TestInferenceAPI(unittest.TestCase):
    def setUp(self):
        # Initialize TestClient
        self.client = TestClient(app)

        self.mock_inference = MagicMock(spec=Inference)
        # Patch the _config dependency
        patcher = patch('marqo.inference.native_inference.remote.server.inference_api._config')
        self.mock_config = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_config.local_inference = self.mock_inference

    def test_vectorise_success(self):
        self.mock_inference.vectorise.side_effect = [InferenceResult(result=[[('chunk', np.array([0.1, 0.2, 0.3]))]])]

        # Prepare a valid InferenceRequest
        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            model_config=ModelConfig(model_name='random'),
            preprocessing_config=TextPreprocessingConfig()
        )
        packed_data = msgpack.packb(inference_request.dict(), use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            data=packed_data
        )

        self.assertEqual(response.status_code, HTTP_200_OK)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("result", unpacked_response)
        self.assertEqual(1, len(unpacked_response["result"]))  # result for one content
        self.assertEqual(1, len(unpacked_response["result"][0]))  # only one chunk
        self.assertEqual(2, len(unpacked_response["result"][0][0]))  # two elements, chunk key and embeddings
        self.assertEqual("chunk", unpacked_response["result"][0][0][0])  # first element of first chunk
        self.assertTrue(np.array_equal([0.1, 0.2, 0.3], unpacked_response["result"][0][0][1]))
        self.mock_inference.vectorise.assert_called_once_with(inference_request)

    def test_vectorise_invalid_msgpack_request(self):
        # Send invalid MessagePack data
        invalid_data = b'not a valid msgpack'

        response = self.client.post(
            "/vectorise",
            headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            data=invalid_data
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
            headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            data=packed_data
        )

        self.assertEqual(response.status_code, HTTP_422_UNPROCESSABLE_ENTITY)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertEquals({'loc': ['modality'], 'msg': 'field required', 'type': 'value_error.missing'},
                          unpacked_response["detail"][0])

    def test_vectorise_raise_inference_error(self):
        # Configure the mock to raise an Exception
        self.mock_inference.vectorise.side_effect = InferenceError("Inference failed")

        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            model_config=ModelConfig(model_name='random'),
            preprocessing_config=TextPreprocessingConfig()
        )
        packed_data = msgpack.packb(inference_request.dict(), use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            data=packed_data
        )

        self.assertEqual(response.status_code, HTTP_400_BAD_REQUEST)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("An error occurred during vectorisation. Inference failed", unpacked_response["detail"])

    def test_vectorise_raise_exception_other_than_inference_error(self):
        self.mock_inference.vectorise.side_effect = ValueError("Some internal error")

        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            model_config=ModelConfig(model_name='random'),
            preprocessing_config=TextPreprocessingConfig()
        )
        packed_data = msgpack.packb(inference_request.dict(), use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            data=packed_data
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("Some internal error", unpacked_response["detail"])

    def test_vectorise_unsupported_content_type(self):
        inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            model_config=ModelConfig(model_name='random'),
            preprocessing_config=TextPreprocessingConfig()
        )
        packed_data = msgpack.packb(inference_request.dict(), use_bin_type=True)

        response = self.client.post(
            "/vectorise",
            headers={"Content-Type": "application/protobuf", "Accept": "application/msgpack"},
            data=packed_data
        )

        self.assertEqual(response.status_code, HTTP_415_UNSUPPORTED_MEDIA_TYPE)
        unpacked_response = msgpack.unpackb(response.content, raw=False)
        self.assertIn("detail", unpacked_response)
        self.assertIn("Unsupported Content-Type", unpacked_response["detail"])

    def test_healthz_happy_pass(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @unittest.skip(reason='not supported yet')
    def test_healthz_fails_if_exception_raised(self):
        for cuda_exception in [
            CudaDeviceNotAvailableError('CUDA device(s) have become unavailable'),
            CudaOutOfMemoryError('CUDA device cuda:0(Tesla T4) is out of memory')
        ]:
            with self.subTest(cuda_exception):
                with patch("marqo.core.inference.device_manager.DeviceManager.cuda_device_health_check",
                           side_effect=cuda_exception):
                    response = self.client.get("/healthz")
                    self.assertEqual(response.status_code, 503)
                    self.assertIn(cuda_exception.message, response.json()['message'])

