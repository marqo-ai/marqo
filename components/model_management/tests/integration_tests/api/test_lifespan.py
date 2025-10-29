"""Integration tests for application lifespan management."""

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_management.api.lifespan import lifespan


class TestLifespanIntegration(TestCase):
    """Integration tests for application lifespan functionality."""

    def test_lifespan_initializes_settings(self):
        """Test that lifespan initializes settings on startup."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "json"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch("model_management.api.lifespan.on_start"):
                        with TestClient(app):
                            mock_get_settings.assert_called()

    def test_lifespan_initializes_logger(self):
        """Test that lifespan initializes logger with correct settings."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "json"
            mock_settings.marqo_log_level = "debug"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch(
                "model_management.api.lifespan.instantiate_logger"
            ) as mock_instantiate_logger:
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch("model_management.api.lifespan.on_start"):
                        with TestClient(app):
                            mock_instantiate_logger.assert_called_once_with(
                                mock_settings
                            )

    def test_lifespan_initializes_config(self):
        """Test that lifespan initializes configuration on startup."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch("model_management.api.lifespan.on_start"):
                        with TestClient(app):
                            mock_get_config.assert_called()

    def test_lifespan_runs_on_start_tasks(self):
        """Test that lifespan runs on_start tasks."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch(
                        "model_management.api.lifespan.on_start"
                    ) as mock_on_start:
                        mock_on_start.return_value = None

                        with TestClient(app):
                            mock_on_start.assert_called_once_with(
                                mock_config, mock_settings
                            )

    def test_lifespan_handles_async_on_start(self):
        """Test that lifespan handles async on_start tasks."""
        import asyncio

        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    async def async_on_start(cfg, settings):
                        await asyncio.sleep(0.001)
                        return True

                    with patch(
                        "model_management.api.lifespan.on_start",
                        side_effect=async_on_start,
                    ):
                        with TestClient(app):
                            # Should complete without error
                            pass

    def test_lifespan_initialization_order(self):
        """Test that lifespan initializes components in correct order."""
        app = FastAPI(lifespan=lifespan)
        call_order = []

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []

            def track_get_settings():
                call_order.append("get_settings")
                return mock_settings

            mock_get_settings.side_effect = track_get_settings

            with patch(
                "model_management.api.lifespan.instantiate_logger"
            ) as mock_logger:

                def track_instantiate_logger(s):
                    call_order.append("instantiate_logger")

                mock_logger.side_effect = track_instantiate_logger

                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()

                    def track_get_config():
                        call_order.append("get_config")
                        return mock_config

                    mock_get_config.side_effect = track_get_config

                    with patch(
                        "model_management.api.lifespan.on_start"
                    ) as mock_on_start:

                        def track_on_start(cfg, settings):
                            call_order.append("on_start")

                        mock_on_start.side_effect = track_on_start

                        with TestClient(app):
                            pass

        # Verify order: settings -> logger -> config -> on_start
        self.assertEqual(
            ["get_settings", "instantiate_logger", "get_config", "on_start"], call_order
        )

    def test_lifespan_context_manager_yields_control(self):
        """Test that lifespan context manager yields control to application."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch("model_management.api.lifespan.on_start"):
                        # Add a test route
                        @app.get("/test")
                        def test_route():
                            return {"status": "ok"}

                        with TestClient(app) as client:
                            # Application should be running and responding
                            response = client.get("/test")
                            self.assertEqual(200, response.status_code)

    def test_lifespan_with_preload_models_empty(self):
        """Test that lifespan handles empty preload models list."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch(
                        "model_management.api.lifespan.on_start"
                    ) as mock_on_start:
                        with TestClient(app):
                            mock_on_start.assert_called_once()

    def test_lifespan_with_preload_models_not_none(self):
        """Test that lifespan handles preload models configuration."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "text"
            mock_settings.marqo_log_level = "info"
            mock_settings.marqo_models_to_preload = [
                MagicMock(name="model1"),
                MagicMock(name="model2"),
            ]
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_config"
                ) as mock_get_config:
                    mock_config = MagicMock()
                    mock_get_config.return_value = mock_config

                    with patch(
                        "model_management.api.lifespan.on_start"
                    ) as mock_on_start:
                        with TestClient(app):
                            mock_on_start.assert_called_once_with(
                                mock_config, mock_settings
                            )

    def test_lifespan_logs_configuration(self):
        """Test that lifespan logs configuration on startup."""
        app = FastAPI(lifespan=lifespan)

        with patch("model_management.api.lifespan.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.marqo_log_format = "json"
            mock_settings.marqo_log_level = "debug"
            mock_settings.marqo_models_to_preload = []
            mock_get_settings.return_value = mock_settings

            with patch("model_management.api.lifespan.instantiate_logger"):
                with patch(
                    "model_management.api.lifespan.get_logger"
                ) as mock_get_logger:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger

                    with patch(
                        "model_management.api.lifespan.get_config"
                    ) as mock_get_config:
                        mock_config = MagicMock()
                        mock_get_config.return_value = mock_config

                        with patch("model_management.api.lifespan.on_start"):
                            with TestClient(app):
                                # Verify logger.info was called with configuration
                                mock_logger.info.assert_called_once()
                                call_args = mock_logger.info.call_args
                                self.assertIn("Logger configured", call_args[0][0])


class TestLifespanWithMainApp(TestCase):
    """Test lifespan integration with the main application."""

    def test_main_app_starts_successfully(self):
        """Test that the main application starts successfully with lifespan."""
        # Mock the dependencies to avoid loading real models
        with patch("model_management.config.ModelManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            with patch("model_management.config.TritonClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                # Set environment variables to control startup behavior
                with patch.dict(
                    os.environ,
                    {
                        "MARQO_MODELS_TO_PRELOAD": "[]",
                        "MARQO_LOG_LEVEL": "info",
                        "MARQO_LOG_FORMAT": "text",
                    },
                ):
                    from model_management.main import app

                    with TestClient(app) as client:
                        # Application should start and be responsive
                        response = client.get("/v1/healthz")
                        self.assertEqual(200, response.status_code)

    def test_main_app_healthz_available_after_startup(self):
        """Test that healthz endpoint is available after lifespan startup."""
        with patch("model_management.config.ModelManager"):
            with patch("model_management.config.TritonClient"):
                with patch.dict(
                    os.environ,
                    {"MARQO_MODELS_TO_PRELOAD": "[]", "MARQO_LOG_LEVEL": "error"},
                ):
                    from model_management.main import app

                    with TestClient(app) as client:
                        response = client.get("/v1/healthz")
                        self.assertEqual(200, response.status_code)
                        self.assertEqual({"status": "ok"}, response.json())

    def test_main_app_api_routes_available_after_startup(self):
        """Test that API routes are available after lifespan startup."""
        with patch("model_management.config.ModelManager"):
            with patch("model_management.config.TritonClient"):
                with patch.dict(os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]"}):
                    from model_management.main import app

                    with TestClient(app) as client:
                        # Test unload endpoint (doesn't require model loading)
                        response = client.post("/v1/models/test-model/unload")
                        self.assertEqual(200, response.status_code)

    def test_main_app_openapi_schema_available_after_startup(self):
        """Test that OpenAPI schema is available after lifespan startup."""
        with patch("model_management.config.ModelManager"):
            with patch("model_management.config.TritonClient"):
                with patch.dict(os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]"}):
                    from model_management.main import app

                    with TestClient(app) as client:
                        response = client.get("/openapi.json")
                        self.assertEqual(200, response.status_code)

                        schema = response.json()
                        self.assertIn("info", schema)
                        self.assertIn("paths", schema)

    def test_main_app_handles_startup_with_different_log_levels(self):
        """Test that application starts with different log levels."""
        log_levels = ["debug", "info", "warning", "error"]

        for log_level in log_levels:
            with self.subTest(log_level=log_level):
                with patch("model_management.config.ModelManager"):
                    with patch("model_management.config.TritonClient"):
                        with patch.dict(
                            os.environ,
                            {
                                "MARQO_MODELS_TO_PRELOAD": "[]",
                                "MARQO_LOG_LEVEL": log_level,
                            },
                        ):
                            from model_management.main import app

                            with TestClient(app) as client:
                                response = client.get("/v1/healthz")
                                self.assertEqual(200, response.status_code)

    def test_main_app_handles_startup_with_different_log_formats(self):
        """Test that application starts with different log formats."""
        log_formats = ["text", "json"]

        for log_format in log_formats:
            with self.subTest(log_format=log_format):
                with patch("model_management.config.ModelManager"):
                    with patch("model_management.config.TritonClient"):
                        with patch.dict(
                            os.environ,
                            {
                                "MARQO_MODELS_TO_PRELOAD": "[]",
                                "MARQO_LOG_FORMAT": log_format,
                            },
                        ):
                            from model_management.main import app

                            with TestClient(app) as client:
                                response = client.get("/v1/healthz")
                                self.assertEqual(200, response.status_code)

    def test_lifespan_cleanup_on_shutdown(self):
        """Test that lifespan properly cleans up on application shutdown."""
        with patch("model_management.config.ModelManager"):
            with patch("model_management.config.TritonClient"):
                with patch.dict(os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]"}):
                    from model_management.main import app

                    with TestClient(app) as client:
                        # Make a request to ensure app is running
                        response = client.get("/v1/healthz")
                        self.assertEqual(200, response.status_code)

                    # Context manager exit should trigger shutdown
                    # No exceptions should be raised

    def test_multiple_startup_shutdown_cycles(self):
        """Test that application can go through multiple startup/shutdown cycles."""
        with patch("model_management.config.ModelManager"):
            with patch("model_management.config.TritonClient"):
                with patch.dict(os.environ, {"MARQO_MODELS_TO_PRELOAD": "[]"}):
                    from model_management.main import app

                    for _ in range(3):
                        with TestClient(app) as client:
                            response = client.get("/v1/healthz")
                            self.assertEqual(200, response.status_code)
