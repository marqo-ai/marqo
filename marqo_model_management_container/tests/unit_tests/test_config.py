from unittest import TestCase
from unittest.mock import patch, MagicMock

from marqo_model_management_container.config import Config, get_config
from marqo_model_management_container.core.settings import Settings
from marqo_model_management_container.services.triton.triton_client import TritonClient
from marqo_model_management_container.services.model_manager.model_manager import ModelManager


class TestConfig(TestCase):
    """Test class for Config in marqo_model_management_container.config."""

    def test_config_initialization_with_default_settings(self):
        """Test that Config initializes correctly with default Settings."""
        with patch("os.environ", {}):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(settings.model_base_dir, config.model_manager.model_base_dir)
            self.assertEqual(settings.triton_url, config.triton_client.url)

    def test_config_initialization_with_custom_settings(self):
        """Test that Config initializes correctly with custom Settings."""
        custom_env = {
            "TRITON_URL": "http://custom-triton:9000",
            "MODEL_BASE_DIR": "/custom/models",
        }

        with patch("os.environ", custom_env):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual("/custom/models", config.model_manager.model_base_dir)
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
                with patch("os.environ", {"TRITON_URL": url}):
                    settings = Settings(_env_file=None)

                    with patch("marqo_model_management_container.config.TritonClient") as mock_triton_client:
                        config = Config(settings)
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
                with patch("os.environ", {"MODEL_BASE_DIR": path}):
                    settings = Settings(_env_file=None)

                    with patch("marqo_model_management_container.config.TritonClient") as mock_triton_client, \
                         patch("marqo_model_management_container.config.ModelManager") as mock_model_manager:

                        mock_triton_instance = MagicMock()
                        mock_triton_client.return_value = mock_triton_instance

                        config = Config(settings)

                        mock_model_manager.assert_called_once_with(
                            model_base_dir=path,
                            triton_client=mock_triton_instance
                        )

    def test_get_config_returns_config_instance(self):
        """Test that get_config() returns a Config instance."""
        with patch("os.environ", {}):
            with patch("marqo_model_management_container.config.get_settings") as mock_get_settings:
                mock_settings = Settings(_env_file=None)
                mock_get_settings.return_value = mock_settings

                config = get_config()

                self.assertIsInstance(config, Config)
                mock_get_settings.assert_called_once()

    def test_get_config_uses_get_settings(self):
        """Test that get_config() calls get_settings() to retrieve settings."""
        with patch("marqo_model_management_container.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock(spec=Settings)
            mock_settings.triton_url = "http://localhost:8000"
            mock_settings.model_base_dir = "./cache/models"
            mock_get_settings.return_value = mock_settings

            with patch("marqo_model_management_container.config.TritonClient"), \
                 patch("marqo_model_management_container.config.ModelManager"):
                config = get_config()

                mock_get_settings.assert_called_once()

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
        import json

        model = {
            "maxBatchSize": 8,
            "name": "test-model",
            "sources": ["s3://test/model.onnx"],
            "input": [{"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}]
        }

        with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": json.dumps([model])}):
            settings = Settings(_env_file=None)
            config = Config(settings)

            self.assertIsInstance(config.triton_client, TritonClient)
            self.assertIsInstance(config.model_manager, ModelManager)
            self.assertEqual(1, len(settings.marqo_models_to_preload))
            self.assertEqual("test-model", settings.marqo_models_to_preload[0].name)
            self.assertEqual(8, settings.marqo_models_to_preload[0].max_batch_size)