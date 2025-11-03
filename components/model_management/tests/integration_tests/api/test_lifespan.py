"""Integration tests for application lifespan management."""

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_management.api.lifespan import lifespan


class TestLifespanIntegration(TestCase):
    """Integration tests for application lifespan functionality."""

    def _create_mock_settings(
        self, log_format="text", log_level="info", preload_models=None
    ):
        """Helper to create mock settings."""
        mock_settings = MagicMock()
        mock_settings.marqo_log_format = log_format
        mock_settings.marqo_log_level = log_level
        mock_settings.marqo_models_to_preload = (
            preload_models if preload_models is not None else []
        )
        return mock_settings

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_initialization_components(
        self, mock_get_settings, mock_instantiate_logger, mock_get_config, mock_on_start
    ):
        """Test that lifespan initializes all components correctly."""
        mock_settings = self._create_mock_settings()
        mock_get_settings.return_value = mock_settings
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        app = FastAPI(lifespan=lifespan)
        with TestClient(app):
            mock_get_settings.assert_called()
            mock_instantiate_logger.assert_called_once_with(mock_settings)
            mock_get_config.assert_called()
            mock_on_start.assert_called_once_with(mock_config, mock_settings)

    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_handles_async_on_start(
        self, mock_get_settings, mock_instantiate_logger, mock_get_config
    ):
        """Test that lifespan handles async on_start tasks."""
        import asyncio

        mock_settings = self._create_mock_settings()
        mock_get_settings.return_value = mock_settings
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        async def async_on_start(cfg, settings):
            await asyncio.sleep(0.001)
            return True

        app = FastAPI(lifespan=lifespan)
        with patch(
            "model_management.api.lifespan.on_start", side_effect=async_on_start
        ):
            with TestClient(app):
                pass  # Should complete without error

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_initialization_order(
        self, mock_get_settings, mock_logger, mock_get_config, mock_on_start
    ):
        """Test that lifespan initializes components in correct order."""
        call_order = []
        mock_settings = self._create_mock_settings()

        mock_get_settings.side_effect = lambda: (
            call_order.append("get_settings"),
            mock_settings,
        )[1]
        mock_logger.side_effect = lambda s: call_order.append("instantiate_logger")
        mock_config = MagicMock()
        mock_get_config.side_effect = lambda: (
            call_order.append("get_config"),
            mock_config,
        )[1]
        mock_on_start.side_effect = lambda cfg, settings: call_order.append("on_start")

        app = FastAPI(lifespan=lifespan)
        with TestClient(app):
            pass

        self.assertEqual(
            ["get_settings", "instantiate_logger", "get_config", "on_start"], call_order
        )

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_logs_configuration(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan logs configuration on startup."""
        mock_settings = self._create_mock_settings(log_format="json", log_level="debug")
        mock_get_settings.return_value = mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        app = FastAPI(lifespan=lifespan)
        with TestClient(app):
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            self.assertIn("Logger configured", call_args[0][0])


class TestLifespanWithMainApp(TestCase):
    """Test lifespan integration with the main application."""

    @patch("model_management.config.TritonClient")
    @patch("model_management.config.ModelManager")
    def test_main_app_endpoints_available_after_startup(
        self, mock_manager_class, mock_client_class
    ):
        """Test that main application starts successfully and endpoints are available."""
        with patch.dict(
            os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]", "MARQO_LOG_LEVEL": "error"}
        ):
            from model_management.main import app

            with TestClient(app) as client:
                # Test healthz endpoint
                response = client.get("/v1/healthz")
                self.assertEqual(200, response.status_code)
                self.assertEqual({"status": "ok"}, response.json())

                # Test OpenAPI schema available
                response = client.get("/openapi.json")
                self.assertEqual(200, response.status_code)
                schema = response.json()
                self.assertIn("info", schema)
                self.assertIn("paths", schema)

    @patch("model_management.config.TritonClient")
    @patch("model_management.config.ModelManager")
    def test_main_app_handles_different_logging_configurations(
        self, mock_manager_class, mock_client_class
    ):
        """Test that application starts with different logging configurations."""
        test_cases = [
            ("debug", "text"),
            ("info", "json"),
            ("warning", "text"),
            ("error", "json"),
        ]

        for log_level, log_format in test_cases:
            with self.subTest(log_level=log_level, log_format=log_format):
                with patch.dict(
                    os.environ,
                    {
                        "MARQO_MODELS_TO_PRELOAD": "[]",
                        "MARQO_LOG_LEVEL": log_level,
                        "MARQO_LOG_FORMAT": log_format,
                    },
                ):
                    from model_management.main import app

                    with TestClient(app) as client:
                        response = client.get("/v1/healthz")
                        self.assertEqual(200, response.status_code)

    @patch("model_management.config.TritonClient")
    @patch("model_management.config.ModelManager")
    def test_main_app_lifecycle(self, mock_manager_class, mock_client_class):
        """Test that application handles startup/shutdown cycles correctly."""
        with patch.dict(os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]"}):
            from model_management.main import app

            # Test multiple startup/shutdown cycles
            for _ in range(3):
                with TestClient(app) as client:
                    response = client.get("/v1/healthz")
                    self.assertEqual(200, response.status_code)
                # Context manager exit triggers shutdown without exceptions
