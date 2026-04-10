"""
Integration tests for ModelManagementClient.

This module tests the ModelManagementClient class which communicates with the
model management container to load/unload models in Triton Inference Server.
"""

import unittest
from unittest.mock import MagicMock, patch

import httpx

from inference_orchestrator.services.errors import (
    ModelManagementServiceUnavailableError,
    TritonModelLoadError,
)
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)


class TestModelManagementClient(unittest.TestCase):
    """Test suite for ModelManagementClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://localhost:8080"
        self.client = ModelManagementClient(self.base_url)

    def test_client_initialization(self):
        """Test ModelManagementClient initializes correctly."""
        client = ModelManagementClient("http://test:9000")
        self.assertEqual(client.url, "http://test:9000")
        self.assertIsInstance(client.client, httpx.Client)

    @patch("httpx.Client.post")
    def test_load_model_success(self, mock_post):
        """Test successful model loading."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model_properties = {
            "modelName": "test_model",
            "modelVersion": "1",
        }

        # Should not raise any exception
        self.client.load_model(model_properties)

        mock_post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/load",
            json={"tritonModelProperties": model_properties},
            timeout=600,
        )

    @patch("httpx.Client.post")
    def test_load_model_with_custom_timeout(self, mock_post):
        """Test model loading with custom timeout."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model_properties = {"modelName": "test_model"}
        custom_timeout = 300

        self.client.load_model(model_properties, timeout=custom_timeout)

        mock_post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/load",
            json={"tritonModelProperties": model_properties},
            timeout=custom_timeout,
        )

    @patch("httpx.Client.post")
    def test_load_model_connect_error(self, mock_post):
        """Test load_model raises ModelManagementServiceUnavailableError on ConnectError."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        model_properties = {"modelName": "test_model"}

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.load_model(model_properties)

        self.assertIn("unavailable", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_load_model_network_error(self, mock_post):
        """Test load_model raises ModelManagementServiceUnavailableError on NetworkError."""
        mock_post.side_effect = httpx.NetworkError("Network error")

        model_properties = {"modelName": "test_model"}

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.load_model(model_properties)

        self.assertIn("unavailable", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_load_model_timeout_error(self, mock_post):
        """Test load_model raises ModelManagementServiceUnavailableError on TimeoutException."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        model_properties = {"modelName": "test_model"}

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.load_model(model_properties)

        self.assertIn("timed out", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_load_model_http_status_error(self, mock_post):
        """Test load_model raises TritonModelLoadError on HTTP status error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_response

        model_properties = {"modelName": "test_model"}

        with self.assertRaises(TritonModelLoadError) as context:
            self.client.load_model(model_properties)

        self.assertIn("Failed to load", str(context.exception))

    @patch("httpx.Client.post")
    def test_unload_model_success(self, mock_post):
        """Test successful model unloading."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model_name = "test_model"

        # Should not raise any exception
        self.client.unload_model(model_name)

        mock_post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/{model_name}/unload?remove-files=False",
            timeout=60,
        )

    @patch("httpx.Client.post")
    def test_unload_model_with_remove_files(self, mock_post):
        """Test model unloading with remove_files=True."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model_name = "test_model"

        self.client.unload_model(model_name, remove_files=True)

        mock_post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/{model_name}/unload?remove-files=True",
            timeout=60,
        )

    @patch("httpx.Client.post")
    def test_unload_model_with_custom_timeout(self, mock_post):
        """Test model unloading with custom timeout."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model_name = "test_model"
        custom_timeout = 120

        self.client.unload_model(model_name, timeout=custom_timeout)

        mock_post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/{model_name}/unload?remove-files=False",
            timeout=custom_timeout,
        )

    @patch("httpx.Client.post")
    def test_unload_model_connect_error(self, mock_post):
        """Test unload_model raises ModelManagementServiceUnavailableError on ConnectError."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.unload_model("test_model")

        self.assertIn("unavailable", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_unload_model_network_error(self, mock_post):
        """Test unload_model raises ModelManagementServiceUnavailableError on NetworkError."""
        mock_post.side_effect = httpx.NetworkError("Network error")

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.unload_model("test_model")

        self.assertIn("unavailable", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_unload_model_timeout_error(self, mock_post):
        """Test unload_model raises ModelManagementServiceUnavailableError on TimeoutException."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            self.client.unload_model("test_model")

        self.assertIn("timed out", str(context.exception).lower())

    @patch("httpx.Client.post")
    def test_unload_model_http_status_error(self, mock_post):
        """Test unload_model raises TritonModelLoadError on HTTP status error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_response

        with self.assertRaises(TritonModelLoadError) as context:
            self.client.unload_model("test_model")

        self.assertIn("Failed to unload", str(context.exception))


if __name__ == "__main__":
    unittest.main()
