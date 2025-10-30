import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

from model_management.config import Config
from model_management.core.settings import Settings
from model_management.services.model_manager.model_manager import ModelManager
from model_management.services.triton.triton_client import TritonClient


class TestConfig(TestCase):
    """Test class for Config in model_management.config."""

    def test_config_initialization_with_default_settings(self):
        """Test that Config initializes correctly with default Settings."""
        with patch("os.environ", {}):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(
                settings.marqo_model_cache_path,
                config.model_manager.marqo_model_cache_path,
            )
            self.assertEqual(settings.marqo_triton_rest_url, config.triton_client.url)

    def test_config_initialization_with_custom_settings(self):
        """Test that Config initializes correctly with custom Settings."""
        custom_env = {
            "MARQO_TRITON_REST_URL": "http://custom-triton:9000",
            "MARQO_MODEL_CACHE_PATH": "/custom/models",
        }

        with patch("os.environ", custom_env):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(
                "/custom/models", config.model_manager.marqo_model_cache_path
            )
            self.assertEqual("http://custom-triton:9000", config.triton_client.url)
            self.assertIs(config.model_manager.triton_client, config.triton_client)

    def test_config_triton_client_receives_correct_url(self):
        """Test that TritonClient is initialized with the correct URL from Settings."""
        test_cases = [
            ("http://localhost:8000", "default URL"),
            ("http://triton-service:8000", "custom URL"),
            ("https://secure-triton:8443", "HTTPS URL"),
        ]

        for url, msg in test_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_TRITON_REST_URL": url}):
                    settings = Settings(_env_file=None)

                    with patch(
                        "model_management.config.TritonClient"
                    ) as mock_triton_client:
                        _ = Config(settings)
                        mock_triton_client.assert_called_once_with(url=url)

    def test_config_model_manager_receives_correct_parameters(self):
        """Test that ModelManager is initialized with correct parameters from Settings."""
        test_cases = [
            ("./cache/models", "default path"),
            ("/tmp/models", "absolute path"),
            ("~/models", "home directory path"),
        ]

        for path, msg in test_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_MODEL_CACHE_PATH": path}):
                    settings = Settings(_env_file=None)

                    with (
                        patch(
                            "model_management.config.TritonClient"
                        ) as mock_triton_client,
                        patch(
                            "model_management.config.ModelManager"
                        ) as mock_model_manager,
                    ):
                        mock_triton_instance = MagicMock()
                        mock_triton_client.return_value = mock_triton_instance

                        _ = Config(settings)

                        mock_model_manager.assert_called_once_with(
                            marqo_model_cache_path=path,
                            triton_client=mock_triton_instance,
                        )

    def test_config_components_are_accessible(self):
        """Test that Config components (triton_client, model_manager) are accessible and of correct type."""
        with patch("os.environ", {}):
            settings = Settings(_env_file=None)
            config = Config(settings)

            # Verify components exist, are accessible, and are of the correct type
            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(type(config.triton_client).__name__, "TritonClient")
            self.assertEqual(type(config.model_manager).__name__, "ModelManager")

    def test_config_with_models_to_preload(self):
        """Test that Config initializes correctly when Settings has models_to_preload."""
        model = {
            "maxBatchSize": 8,
            "name": "test-model",
            "sources": ["s3://test/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        }

        with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": json.dumps([model])}):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(1, len(settings.marqo_models_to_preload))
            self.assertEqual("test-model", settings.marqo_models_to_preload[0].name)
            self.assertEqual(8, settings.marqo_models_to_preload[0].max_batch_size)
