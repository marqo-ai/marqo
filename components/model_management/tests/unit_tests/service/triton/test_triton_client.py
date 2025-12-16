from unittest import TestCase
from unittest.mock import Mock, patch

import httpx
from httpx import ConnectError, HTTPStatusError, NetworkError, TimeoutException
from model_management.services.errors import (
    TritonCommunicationError,
    TritonModelLoadError,
)
from model_management.services.triton.triton_client import TritonClient


class TestTritonClient(TestCase):
    """Test class for TritonClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.url = "http://localhost:8000"
        self.client = TritonClient(url=self.url)

    def test_triton_client_initialization(self):
        """Test TritonClient initialization."""
        client = TritonClient(url="http://triton:8000")

        self.assertEqual("http://triton:8000", client.url)
        self.assertIsInstance(client.client, httpx.Client)

    def test_triton_client_initialization_with_various_urls(self):
        """Test TritonClient initialization with various URL formats."""
        test_cases = [
            "http://localhost:8000",
            "http://triton-service:8000",
            "https://triton.example.com:8443",
            "http://192.168.1.100:8000",
        ]

        for url in test_cases:
            with self.subTest(url=url):
                client = TritonClient(url=url)
                self.assertEqual(url, client.url)

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_success(self, mock_client_class):
        """Test successful model loading."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)
        client.load_model("test-model")

        mock_client_instance.post.assert_called_once_with(
            f"{self.url}/v2/repository/models/test-model/load",
            timeout=httpx.Timeout(5, read=30),
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_timeout(self, mock_client_class):
        """Test load_model raises TritonCommunicationError on timeout."""
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(side_effect=TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonCommunicationError) as context:
            client.load_model("test-model")

        self.assertIn("Triton timed out", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_connection_error(self, mock_client_class):
        """Test load_model raises TritonCommunicationError on connection error."""
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(side_effect=ConnectError("Connection failed"))
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonCommunicationError) as context:
            client.load_model("test-model")

        self.assertIn("Triton is unavailable", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_network_error(self, mock_client_class):
        """Test load_model raises TritonCommunicationError on network error."""
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(side_effect=NetworkError("Network error"))
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonCommunicationError) as context:
            client.load_model("test-model")

        self.assertIn("Triton is unavailable", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_http_status_error(self, mock_client_class):
        """Test load_model raises TritonModelLoadError on HTTP status error."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={"error": "Model not found"})
        mock_response.raise_for_status = Mock(
            side_effect=HTTPStatusError(
                "400 Bad Request", request=Mock(), response=mock_response
            )
        )
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonModelLoadError) as context:
            client.load_model("test-model")

        self.assertIn("Failed to load model", str(context.exception))
        self.assertIn("Model not found", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_unload_model_success(self, mock_client_class):
        """Test successful model unloading."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)
        client.unload_model("test-model")

        mock_client_instance.post.assert_called_once_with(
            f"{self.url}/v2/repository/models/test-model/unload",
            timeout=httpx.Timeout(5, read=30),
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_unload_model_timeout(self, mock_client_class):
        """Test unload_model raises TritonCommunicationError on timeout."""
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(side_effect=TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonCommunicationError) as context:
            client.unload_model("test-model")

        self.assertIn("Triton timed out", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_unload_model_connection_error(self, mock_client_class):
        """Test unload_model raises TritonCommunicationError on connection error."""
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(side_effect=ConnectError("Connection failed"))
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonCommunicationError) as context:
            client.unload_model("test-model")

        self.assertIn("Triton is unavailable", str(context.exception))

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_unload_model_http_status_error(self, mock_client_class):
        """Test unload_model raises TritonModelLoadError on HTTP status error."""
        mock_response = Mock()
        mock_response.json = Mock(return_value={"error": "Model is not loaded"})
        mock_response.raise_for_status = Mock(
            side_effect=HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
        )
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        with self.assertRaises(TritonModelLoadError) as context:
            client.unload_model("test-model")

        self.assertIn("Failed to unload model", str(context.exception))
        self.assertIn("Model is not loaded", str(context.exception))

    def test_get_loaded_models_not_implemented(self):
        """Test that get_loaded_models raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.client.get_loaded_models()

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_load_model_with_various_model_names(self, mock_client_class):
        """Test load_model with various model name formats."""
        test_cases = [
            "simple-model",
            "model_with_underscores",
            "model-123",
            "ModelCamelCase",
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                client.load_model(model_name)
                expected_url = f"{self.url}/v2/repository/models/{model_name}/load"
                call_args = mock_client_instance.post.call_args
                self.assertEqual(expected_url, call_args[0][0])

    @patch("model_management.services.triton.triton_client.httpx.Client")
    def test_timeout_configuration(self, mock_client_class):
        """Test that timeout is configured correctly for requests."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance = Mock()
        mock_client_instance.post = Mock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        client = TritonClient(url=self.url)
        client.load_model("test-model")

        call_args = mock_client_instance.post.call_args
        timeout = call_args[1]["timeout"]
        self.assertIsInstance(timeout, httpx.Timeout)
