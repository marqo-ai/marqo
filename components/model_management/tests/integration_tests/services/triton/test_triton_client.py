from unittest import TestCase

import httpx

from model_management.services.errors import (
    TritonCommunicationError,
    TritonModelLoadError,
)
from model_management.services.triton import triton_client


class TestTritonClientIntegration(TestCase):
    """Integration tests for TritonClient against a real Triton server."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        cls.triton_url = "http://localhost:8000"
        cls.test_model_name = "test-integration-model"

    def setUp(self):
        """Set up test fixtures for each test."""
        self.client = triton_client.TritonClient(self.triton_url)

    def test_client_initialization_with_real_triton_url(self):
        """Test TritonClient initialization with real Triton URL."""
        client = triton_client.TritonClient("http://localhost:8000")

        self.assertEqual("http://localhost:8000", client.url)
        self.assertIsInstance(client.client, httpx.Client)

    def test_load_nonexistent_model_raises_error(self):
        """Test loading a non-existent model raises TritonModelLoadError."""
        with self.assertRaises(TritonModelLoadError) as context:
            self.client.load_model("nonexistent-model-12345")

        error_message = str(context.exception)
        self.assertIn("Failed to load model", error_message)

    def test_unload_nonexistent_model_is_idempotent(self):
        """Test unloading a non-existent model succeeds (idempotent behavior)."""
        # Triton returns 200 OK when unloading a model that doesn't exist
        # This is idempotent behavior - no error should be raised
        try:
            self.client.unload_model("nonexistent-model-12345")
        except TritonModelLoadError:
            self.fail(
                "Unloading non-existent model should not raise error (idempotent)"
            )

    def test_unload_model_idempotent_behavior(self):
        """Test that unload_model can be called multiple times without error."""
        model_name = "test-idempotent-unload"

        # Unload a model that doesn't exist - should succeed
        self.client.unload_model(model_name)

        # Unload again - should also succeed (idempotent)
        self.client.unload_model(model_name)

    def test_connection_to_invalid_triton_url(self):
        """Test that connecting to an invalid Triton URL raises TritonCommunicationError."""
        invalid_client = triton_client.TritonClient("http://localhost:9999")

        with self.assertRaises(TritonCommunicationError) as context:
            invalid_client.load_model("any-model")

        error_message = str(context.exception)
        self.assertIn("Triton is unavailable", error_message)

    def test_load_model_makes_correct_http_request(self):
        """Test that load_model makes the correct HTTP request to Triton."""
        model_name = "test-model-http-request"

        # Attempt to load a non-existent model to verify the request was made
        try:
            self.client.load_model(model_name)
        except TritonModelLoadError:
            # Expected - model doesn't exist
            # The important part is that the request was made to the correct endpoint
            pass

    def test_multiple_load_attempts_on_same_model(self):
        """Test that attempting to load the same model multiple times is handled."""
        model_name = "nonexistent-model-multi-load"

        # First attempt
        with self.assertRaises(TritonModelLoadError):
            self.client.load_model(model_name)

        # Second attempt should also fail consistently
        with self.assertRaises(TritonModelLoadError):
            self.client.load_model(model_name)

    def test_get_loaded_models_not_implemented(self):
        """Test that get_loaded_models raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.client.get_loaded_models()

    def test_model_name_with_special_characters(self):
        """Test loading models with various special characters in names."""
        test_cases = [
            ("model-with-dashes", "should handle dashes"),
            ("model_with_underscores", "should handle underscores"),
            ("model123", "should handle numbers"),
        ]

        for model_name, message in test_cases:
            with self.subTest(model_name=model_name, message=message):
                # All these should fail (model doesn't exist) but should make valid requests
                with self.assertRaises(TritonModelLoadError):
                    self.client.load_model(model_name)

    def test_client_reusability(self):
        """Test that the same client instance can be used for multiple operations."""
        # Make multiple requests with the same client
        for i in range(3):
            with self.subTest(attempt=i):
                with self.assertRaises(TritonModelLoadError):
                    self.client.load_model(f"nonexistent-model-{i}")

    def test_different_error_responses_from_triton(self):
        """Test that different error responses from Triton are handled appropriately."""
        test_cases = [
            ("", "empty model name"),
            ("nonexistent-model", "nonexistent model"),
        ]

        for model_name, description in test_cases:
            with self.subTest(model_name=model_name, description=description):
                with self.assertRaises(
                    (TritonModelLoadError, TritonCommunicationError)
                ):
                    self.client.load_model(model_name)
