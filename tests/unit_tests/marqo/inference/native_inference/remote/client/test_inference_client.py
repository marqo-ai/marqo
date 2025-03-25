# test_inference_client.py
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from marqo.core.inference.api import InferenceRequest, InferenceResult, InferenceError, Modality, ModelConfig, \
    TextPreprocessingConfig
from src.marqo.inference.native_inference.remote.client.inference_client import NativeInferenceClient
import httpx
import msgpack


class TestNativeInferenceClient(unittest.TestCase):
    def setUp(self):
        patcher = patch('src.marqo.inference.native_inference.remote.client.inference_client.httpx.Client')
        self.mock_httpx_client = patcher.start()
        self.addCleanup(patcher.stop)

        self.base_url = "http://mock-inference-service.com"
        self.client = NativeInferenceClient(base_url=self.base_url)

        self.inference_request = InferenceRequest(
            contents=["test content"],
            modality=Modality.TEXT,
            model_config=ModelConfig(model_name='random'),
            preprocessing_config=TextPreprocessingConfig()
        )

    def test_vectorise_successful(self):
        # Prepare mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        result_dict = {'result': [[('chunk1', np.array([1.0, 2.0]))]]}
        mock_response.content = msgpack.packb(result_dict, use_bin_type=True)
        self.mock_httpx_client.return_value.post.return_value = mock_response

        # Call vectorise
        result = self.client.vectorise(self.inference_request)

        # Assertions
        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.result[0][0][0], 'chunk1')
        self.assertTrue(np.array_equal(result.result[0][0][1], np.array([1.0, 2.0])))
        self.mock_httpx_client.return_value.post.assert_called_once()

    def test_vectorise_http_error_with_msgpack(self):
        # Prepare mock HTTPStatusError with msgpack error message
        error_detail = {"detail": "Invalid request"}
        packed_error = msgpack.packb(error_detail, use_bin_type=True)
        mock_response = MagicMock()
        mock_response.content = packed_error
        mock_http_error = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response)
        self.mock_httpx_client.return_value.post.side_effect = mock_http_error

        # Call vectorise and expect InferenceError
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Invalid request", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()

    def test_vectorise_http_error_with_invalid_msgpack(self):
        # Prepare mock HTTPStatusError with invalid msgpack content
        mock_response = MagicMock()
        mock_response.content = b'not msgpack'
        mock_http_error = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response)
        self.mock_httpx_client.return_value.post.side_effect = mock_http_error

        # Call vectorise and expect InferenceError with parsing warning
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Error parsing error message in msgpack format", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()

    def test_vectorise_timeout_error(self):
        # Simulate a timeout error
        self.mock_httpx_client.return_value.post.side_effect = httpx.TimeoutException("Request timed out")

        # Call vectorise and expect InferenceError
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Request timed out", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()

    def test_vectorise_connection_error(self):
        # Simulate a connection error
        self.mock_httpx_client.return_value.post.side_effect = httpx.ConnectError("Connection failed")

        # Call vectorise and expect InferenceError
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Connection failed", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()

    @patch('src.marqo.inference.native_inference.remote.client.inference_client.msgpack.unpackb')
    def test_vectorise_invalid_msgpack_response(self, mock_unpackb):
        # Prepare mock successful response with invalid msgpack
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'invalid msgpack'
        self.mock_httpx_client.return_value.post.return_value = mock_response

        # Configure unpackb to raise an exception
        mock_unpackb.side_effect = msgpack.ExtraData(unpacked=None, extra="Extra data")

        # Call vectorise and expect InferenceError
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Error decoding MessagePack response", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()
        mock_unpackb.assert_called_once_with(mock_response.content, raw=False)

    def test_vectorise_empty_response_content(self):
        # Prepare mock response with empty content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b''
        self.mock_httpx_client.return_value.post.return_value = mock_response

        # Call vectorise and expect InferenceError due to empty response
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("Error decoding MessagePack response", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()

    def test_vectorise_invalid_status_code(self):
        # Prepare mock response with non-200 status code without proper content
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b''
        self.mock_httpx_client.return_value.post.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        # Call vectorise and expect InferenceError
        with self.assertRaises(InferenceError) as context:
            self.client.vectorise(self.inference_request)

        self.assertIn("HTTP error when calling remote inference service", str(context.exception))
        self.mock_httpx_client.return_value.post.assert_called_once()
