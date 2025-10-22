from unittest import TestCase
from unittest.mock import Mock, patch

import httpx

from inference_orchestrator.services.errors import (
    ModelManagementServiceUnavailableError,
    TritonModelLoadError,
)
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)


class TestModelManagementClient(TestCase):
    """Tests for the ModelManagementClient class"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:8000"
        self.client = ModelManagementClient(url=self.base_url)

    def test_init(self):
        """Test that ModelManagementClient initializes correctly"""
        client = ModelManagementClient(url="http://test-server:8000")

        self.assertEqual("http://test-server:8000", client.url)
        self.assertIsInstance(client.client, httpx.Client)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_success(self, mock_httpx_client):
        """Test successful model loading"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)
        model_properties = {
            "name": "test-model",
            "dimensions": 512,
            "type": "open_clip",
        }

        # Should not raise any exception
        client.load_model(model_properties=model_properties, timeout=600)

        # Verify the request was made correctly
        mock_client_instance.post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/load",
            json={"tritonModelProperties": model_properties},
            timeout=600,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_with_custom_timeout(self, mock_httpx_client):
        """Test model loading with custom timeout"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)
        model_properties = {"name": "large-model"}

        client.load_model(model_properties=model_properties, timeout=1200)

        # Verify custom timeout was used
        call_args = mock_client_instance.post.call_args
        self.assertEqual(1200, call_args.kwargs["timeout"])

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_connect_error(self, mock_httpx_client):
        """Test that ConnectError raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.ConnectError("Connection refused")
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.load_model(model_properties={})

        error_message = str(context.exception)
        self.assertIn("unavailable", error_message)
        self.assertIn("Connection refused", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_network_error(self, mock_httpx_client):
        """Test that NetworkError raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.NetworkError(
            "Network unreachable"
        )
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.load_model(model_properties={})

        error_message = str(context.exception)
        self.assertIn("unavailable", error_message)
        self.assertIn("Network unreachable", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_timeout_error(self, mock_httpx_client):
        """Test that TimeoutException raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.TimeoutException(
            "Request timed out"
        )
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.load_model(model_properties={})

        error_message = str(context.exception)
        self.assertIn("timed out", error_message)
        self.assertIn("Request timed out", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_http_status_error(self, mock_httpx_client):
        """Test that HTTPStatusError raises TritonModelLoadError"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request",
            request=Mock(),
            response=Mock(),
        )
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)
        model_properties = {"name": "invalid-model"}

        with self.assertRaises(TritonModelLoadError) as context:
            client.load_model(model_properties=model_properties)

        error_message = str(context.exception)
        self.assertIn("Failed to load", error_message)
        self.assertIn("invalid-model", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_success(self, mock_httpx_client):
        """Test successful model unloading"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        # Should not raise any exception
        client.unload_model(model_name="test-model", remove_files=False, timeout=60)

        # Verify the request was made correctly
        mock_client_instance.post.assert_called_once_with(
            url=f"{self.base_url}/v1/models/test-model/unload?remove-files=False",
            timeout=60,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_with_remove_files(self, mock_httpx_client):
        """Test model unloading with remove_files=True"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        client.unload_model(model_name="test-model", remove_files=True)

        # Verify remove_files parameter was passed correctly
        call_args = mock_client_instance.post.call_args
        self.assertIn("remove-files=True", call_args.kwargs["url"])

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_with_custom_timeout(self, mock_httpx_client):
        """Test model unloading with custom timeout"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        client.unload_model(model_name="test-model", timeout=120)

        # Verify custom timeout was used
        call_args = mock_client_instance.post.call_args
        self.assertEqual(120, call_args.kwargs["timeout"])

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_connect_error(self, mock_httpx_client):
        """Test that ConnectError during unload raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.ConnectError("Connection refused")
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.unload_model(model_name="test-model")

        error_message = str(context.exception)
        self.assertIn("unavailable", error_message)
        self.assertIn("Connection refused", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_network_error(self, mock_httpx_client):
        """Test that NetworkError during unload raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.NetworkError(
            "Network unreachable"
        )
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.unload_model(model_name="test-model")

        error_message = str(context.exception)
        self.assertIn("unavailable", error_message)
        self.assertIn("Network unreachable", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_timeout_error(self, mock_httpx_client):
        """Test that TimeoutException during unload raises ModelManagementServiceUnavailableError"""
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.TimeoutException(
            "Request timed out"
        )
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(ModelManagementServiceUnavailableError) as context:
            client.unload_model(model_name="test-model")

        error_message = str(context.exception)
        self.assertIn("timed out", error_message)
        self.assertIn("Request timed out", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_http_status_error(self, mock_httpx_client):
        """Test that HTTPStatusError during unload raises TritonModelLoadError"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=Mock(),
            response=Mock(),
        )
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        with self.assertRaises(TritonModelLoadError) as context:
            client.unload_model(model_name="nonexistent-model")

        error_message = str(context.exception)
        self.assertIn("Failed to unload", error_message)
        self.assertIn("nonexistent-model", error_message)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_load_model_payload_structure(self, mock_httpx_client):
        """Test that load_model sends the correct payload structure"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)
        model_properties = {
            "name": "test-model",
            "dimensions": 768,
            "type": "hugging_face",
            "url": "hf://test/model",
        }

        client.load_model(model_properties=model_properties)

        # Verify payload has the correct structure
        call_args = mock_client_instance.post.call_args
        payload = call_args.kwargs["json"]

        self.assertIn("tritonModelProperties", payload)
        self.assertEqual(model_properties, payload["tritonModelProperties"])

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_management_client.httpx.Client"
    )
    def test_unload_model_url_format(self, mock_httpx_client):
        """Test that unload_model constructs the URL correctly"""
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        client = ModelManagementClient(url=self.base_url)

        test_cases = [
            (
                "simple-model",
                False,
                f"{self.base_url}/v1/models/simple-model/unload?remove-files=False",
            ),
            (
                "simple-model",
                True,
                f"{self.base_url}/v1/models/simple-model/unload?remove-files=True",
            ),
            (
                "model-with-dashes",
                False,
                f"{self.base_url}/v1/models/model-with-dashes/unload?remove-files=False",
            ),
        ]

        for model_name, remove_files, expected_url in test_cases:
            with self.subTest(model_name=model_name, remove_files=remove_files):
                mock_client_instance.post.reset_mock()

                client.unload_model(model_name=model_name, remove_files=remove_files)

                call_args = mock_client_instance.post.call_args
                self.assertEqual(expected_url, call_args.kwargs["url"])
