from unittest import TestCase
from unittest.mock import Mock, patch

from tritonclient import grpc

from inference_orchestrator.schemas.triton_channel_args import (
    GRPCCompressionAlgorithm,
    TritonChannelArgs,
)
from inference_orchestrator.services.errors import TritonInferenceError
from inference_orchestrator.services.triton_inference.triton.triton_grpc_client import (
    TritonGRPCClient,
)


class TestTritonGRPCClient(TestCase):
    """Tests for the TritonGRPCClient class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_channel_args = Mock(spec=TritonChannelArgs)
        self.mock_channel_args.build_channel_args.return_value = [
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
        self.mock_channel_args.grpc_compression_algorithm = None

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_init_success(self, mock_inference_server_client):
        """Test successful initialization of TritonGRPCClient"""
        url = "http://localhost:8001"

        client = TritonGRPCClient(url=url, triton_channel_args=self.mock_channel_args)

        # Verify URL was parsed correctly (http:// removed)
        mock_inference_server_client.assert_called_once_with(
            url="localhost:8001",
            verbose=False,
            channel_args=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )

        # Verify client was set
        self.assertEqual(mock_inference_server_client.return_value, client.client)

        # Verify compression algorithm was set
        self.assertIsNone(client.grpc_compression_algorithm)

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_init_with_compression(self, mock_inference_server_client):
        """Test initialization with GZIP compression algorithm"""
        url = "https://triton-server:8001"
        self.mock_channel_args.grpc_compression_algorithm = (
            GRPCCompressionAlgorithm.GZIP
        )

        client = TritonGRPCClient(url=url, triton_channel_args=self.mock_channel_args)

        # Verify compression algorithm was set
        self.assertEqual(
            GRPCCompressionAlgorithm.GZIP, client.grpc_compression_algorithm
        )

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_init_with_deflate_compression(self, mock_inference_server_client):
        """Test initialization with DEFLATE compression algorithm"""
        url = "https://triton-server:8001"
        self.mock_channel_args.grpc_compression_algorithm = (
            GRPCCompressionAlgorithm.DEFLATE
        )

        client = TritonGRPCClient(url=url, triton_channel_args=self.mock_channel_args)

        # Verify compression algorithm was set
        self.assertEqual(
            GRPCCompressionAlgorithm.DEFLATE, client.grpc_compression_algorithm
        )

    def test_parse_url_removes_http_prefix(self):
        """Test that _parse_url removes http:// prefix"""
        test_cases = [
            ("http://localhost:8001", "localhost:8001"),
            ("https://localhost:8001", "localhost:8001"),
            ("http://triton-server:8001", "triton-server:8001"),
            ("https://triton-server:8001", "triton-server:8001"),
            ("localhost:8001", "localhost:8001"),
            ("triton-server:8001", "triton-server:8001"),
        ]

        for input_url, expected_output in test_cases:
            with self.subTest(url=input_url):
                with patch(
                    "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
                ):
                    client = TritonGRPCClient(
                        url="http://test:8001",
                        triton_channel_args=self.mock_channel_args,
                    )
                    result = client._parse_url(input_url)
                    self.assertEqual(expected_output, result)

    def test_parse_url_empty_raises_error(self):
        """Test that _parse_url raises ValueError for empty URL"""
        invalid_urls = ["", None]

        for invalid_url in invalid_urls:
            with self.subTest(url=invalid_url):
                with patch(
                    "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
                ):
                    client = TritonGRPCClient(
                        url="http://test:8001",
                        triton_channel_args=self.mock_channel_args,
                    )
                    with self.assertRaises(ValueError) as context:
                        client._parse_url(invalid_url)
                    self.assertIn("cannot be empty", str(context.exception))

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_close_calls_client_close(self, mock_inference_server_client):
        """Test that close() calls the underlying client's close method"""
        mock_client_instance = Mock()
        mock_inference_server_client.return_value = mock_client_instance

        client = TritonGRPCClient(
            url="http://localhost:8001", triton_channel_args=self.mock_channel_args
        )
        client.close()

        # Verify close was called on the underlying client
        mock_client_instance.close.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_encode_success(self, mock_inference_server_client):
        """Test successful encoding with the model"""
        mock_client_instance = Mock()
        mock_inference_server_client.return_value = mock_client_instance

        # Create mock infer result
        mock_infer_result = Mock(spec=grpc.InferResult)
        mock_client_instance.infer.return_value = mock_infer_result

        client = TritonGRPCClient(
            url="http://localhost:8001", triton_channel_args=self.mock_channel_args
        )

        # Create mock inputs and outputs
        mock_inputs = [Mock(spec=grpc.InferInput)]
        mock_outputs = [Mock(spec=grpc.InferRequestedOutput)]

        result = client.encode(
            model_name="test-model",
            infer_inputs=mock_inputs,
            infer_outputs=mock_outputs,
        )

        # Verify infer was called with correct parameters
        mock_client_instance.infer.assert_called_once_with(
            model_name="test-model",
            inputs=mock_inputs,
            outputs=mock_outputs,
            compression_algorithm=None,
        )

        # Verify result is returned
        self.assertEqual(mock_infer_result, result)

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_encode_with_compression(self, mock_inference_server_client):
        """Test encoding with compression algorithm"""
        mock_client_instance = Mock()
        mock_inference_server_client.return_value = mock_client_instance
        mock_infer_result = Mock(spec=grpc.InferResult)
        mock_client_instance.infer.return_value = mock_infer_result

        self.mock_channel_args.grpc_compression_algorithm = (
            GRPCCompressionAlgorithm.GZIP
        )
        client = TritonGRPCClient(
            url="http://localhost:8001", triton_channel_args=self.mock_channel_args
        )

        mock_inputs = [Mock(spec=grpc.InferInput)]
        mock_outputs = [Mock(spec=grpc.InferRequestedOutput)]

        client.encode(
            model_name="test-model",
            infer_inputs=mock_inputs,
            infer_outputs=mock_outputs,
        )

        # Verify compression algorithm was passed
        mock_client_instance.infer.assert_called_once_with(
            model_name="test-model",
            inputs=mock_inputs,
            outputs=mock_outputs,
            compression_algorithm=GRPCCompressionAlgorithm.GZIP,
        )

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_encode_raises_triton_inference_error_on_exception(
        self, mock_inference_server_client
    ):
        """Test that encode raises TritonInferenceError when inference fails"""
        mock_client_instance = Mock()
        mock_inference_server_client.return_value = mock_client_instance

        # Simulate InferenceServerException
        original_error = grpc.InferenceServerException("Model not found")
        mock_client_instance.infer.side_effect = original_error

        client = TritonGRPCClient(
            url="http://localhost:8001", triton_channel_args=self.mock_channel_args
        )

        mock_inputs = [Mock(spec=grpc.InferInput)]
        mock_outputs = [Mock(spec=grpc.InferRequestedOutput)]

        with self.assertRaises(TritonInferenceError) as context:
            client.encode(
                model_name="test-model",
                infer_inputs=mock_inputs,
                infer_outputs=mock_outputs,
            )

        # Verify error message contains model name and original error
        error_message = str(context.exception)
        self.assertIn("test-model", error_message)
        self.assertIn("Model not found", error_message)

        # Verify the exception was chained properly
        self.assertIs(original_error, context.exception.__cause__)

    @patch(
        "inference_orchestrator.services.triton_inference.triton.triton_grpc_client.grpc.InferenceServerClient"
    )
    def test_encode_multiple_inputs_outputs(self, mock_inference_server_client):
        """Test encoding with multiple inputs and outputs"""
        mock_client_instance = Mock()
        mock_inference_server_client.return_value = mock_client_instance
        mock_infer_result = Mock(spec=grpc.InferResult)
        mock_client_instance.infer.return_value = mock_infer_result

        client = TritonGRPCClient(
            url="http://localhost:8001", triton_channel_args=self.mock_channel_args
        )

        # Create multiple mock inputs and outputs
        mock_inputs = [
            Mock(spec=grpc.InferInput),
            Mock(spec=grpc.InferInput),
        ]
        mock_outputs = [
            Mock(spec=grpc.InferRequestedOutput),
            Mock(spec=grpc.InferRequestedOutput),
        ]

        result = client.encode(
            model_name="multi-input-model",
            infer_inputs=mock_inputs,
            infer_outputs=mock_outputs,
        )

        # Verify all inputs and outputs were passed
        call_args = mock_client_instance.infer.call_args
        self.assertEqual(2, len(call_args.kwargs["inputs"]))
        self.assertEqual(2, len(call_args.kwargs["outputs"]))
        self.assertEqual(mock_infer_result, result)
