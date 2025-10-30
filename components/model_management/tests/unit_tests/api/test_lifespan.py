import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from model_management.api.lifespan import lifespan
from model_management.config import Config
from model_management.core.enum import LogFormat, LogLevel
from model_management.core.settings import Settings


class TestLifespan(unittest.TestCase):
    """Test class for FastAPI lifespan context manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = MagicMock(spec=FastAPI)
        self.mock_settings = MagicMock(spec=Settings)
        self.mock_settings.marqo_log_level = LogLevel.INFO
        self.mock_settings.marqo_log_format = LogFormat.PLAIN
        self.mock_config = MagicMock(spec=Config)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_startup_sequence(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan executes startup sequence in correct order."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        mock_on_start.return_value = None

        import asyncio

        async def run_lifespan():
            async with lifespan(self.app):
                pass

        asyncio.run(run_lifespan())

        # Verify the sequence of calls
        mock_get_settings.assert_called_once()
        mock_instantiate_logger.assert_called_once_with(self.mock_settings)
        mock_get_logger.assert_called_once_with("model_management.api.lifespan")
        mock_get_config.assert_called_once()
        mock_on_start.assert_called_once_with(self.mock_config, self.mock_settings)

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
        """Test that lifespan logs the logger configuration."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        mock_on_start.return_value = None

        import asyncio

        async def run_lifespan():
            async with lifespan(self.app):
                pass

        asyncio.run(run_lifespan())

        # Verify logger.info was called with format and level
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        self.assertIn("Logger configured", call_args[0])
        self.assertIn("format", call_args[0])
        self.assertIn("level", call_args[0])
        self.assertEqual(self.mock_settings.marqo_log_format, call_args[1])
        self.assertEqual(self.mock_settings.marqo_log_level, call_args[2])
        self.assertEqual(self.mock_settings.marqo_log_format, call_args[1])
        self.assertEqual(self.mock_settings.marqo_log_level, call_args[2])

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_handles_sync_on_start(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan handles synchronous on_start function."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        # on_start returns None (synchronous)
        mock_on_start.return_value = None

        import asyncio

        async def run_lifespan():
            async with lifespan(self.app):
                pass

        asyncio.run(run_lifespan())

        mock_on_start.assert_called_once_with(self.mock_config, self.mock_settings)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_handles_async_on_start(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan handles asynchronous on_start function."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        # on_start returns an awaitable (asynchronous)
        async_result = AsyncMock()
        mock_on_start.return_value = async_result

        import asyncio

        async def run_lifespan():
            async with lifespan(self.app):
                pass

        asyncio.run(run_lifespan())

        mock_on_start.assert_called_once_with(self.mock_config, self.mock_settings)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_with_different_log_levels(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan works with different log levels."""
        log_levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]

        for log_level in log_levels:
            with self.subTest(log_level=log_level):
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_level = log_level
                mock_settings.marqo_log_format = LogFormat.PLAIN
                mock_get_settings.return_value = mock_settings
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                mock_get_config.return_value = self.mock_config
                mock_on_start.return_value = None

                import asyncio

                async def run_lifespan():
                    async with lifespan(self.app):
                        pass

                asyncio.run(run_lifespan())

                # Verify instantiate_logger was called with correct settings
                mock_instantiate_logger.assert_called_with(mock_settings)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_with_different_marqo_log_formats(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan works with different log formats."""
        marqo_log_formats = [LogFormat.PLAIN, LogFormat.JSON]

        for marqo_log_format in marqo_log_formats:
            with self.subTest(marqo_log_format=marqo_log_format):
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_level = LogLevel.INFO
                mock_settings.marqo_log_format = marqo_log_format
                mock_get_settings.return_value = mock_settings
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                mock_get_config.return_value = self.mock_config
                mock_on_start.return_value = None

                import asyncio

                async def run_lifespan():
                    async with lifespan(self.app):
                        pass

                asyncio.run(run_lifespan())

                # Verify instantiate_logger was called with correct settings
                mock_instantiate_logger.assert_called_with(mock_settings)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_yields_control_during_application_runtime(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan yields control and application can run."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        mock_on_start.return_value = None

        import asyncio

        application_ran = False

        async def run_lifespan():
            nonlocal application_ran
            async with lifespan(self.app):
                # This simulates the application running
                application_ran = True

        asyncio.run(run_lifespan())

        self.assertTrue(application_ran)

    @patch("model_management.api.lifespan.on_start")
    @patch("model_management.api.lifespan.get_config")
    @patch("model_management.api.lifespan.get_logger")
    @patch("model_management.api.lifespan.instantiate_logger")
    @patch("model_management.api.lifespan.get_settings")
    def test_lifespan_completes_without_shutdown_handlers(
        self,
        mock_get_settings,
        mock_instantiate_logger,
        mock_get_logger,
        mock_get_config,
        mock_on_start,
    ):
        """Test that lifespan completes successfully without shutdown handlers."""
        mock_get_settings.return_value = self.mock_settings
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_config.return_value = self.mock_config
        mock_on_start.return_value = None

        import asyncio

        async def run_lifespan():
            async with lifespan(self.app):
                pass
            # Exiting the context should not raise any exceptions

        # Should not raise any exceptions
        asyncio.run(run_lifespan())
