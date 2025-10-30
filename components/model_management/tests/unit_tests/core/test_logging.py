"""Unit tests for logging configuration."""

import logging
import logging.config
from unittest import TestCase
from unittest.mock import MagicMock, patch

from model_management.core.enum import LogFormat, LogLevel
from model_management.core.logging import get_logger, instantiate_logger
from model_management.core.settings import Settings


class TestInstantiateLogger(TestCase):
    """Test class for instantiate_logger function."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset logging configuration before each test
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_configuration_structure(self, mock_dict_config):
        """Test that logger configuration has the correct base structure."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        instantiate_logger(mock_settings)

        mock_dict_config.assert_called_once()
        config = mock_dict_config.call_args[0][0]

        # Verify basic structure
        self.assertEqual(1, config["version"])
        self.assertFalse(config["disable_existing_loggers"])

        # Verify all major sections exist
        self.assertIn("formatters", config)
        self.assertIn("handlers", config)
        self.assertIn("loggers", config)
        self.assertIn("root", config)

        # Verify formatters exist
        self.assertIn("default-plain", config["formatters"])
        self.assertIn("default-json", config["formatters"])
        self.assertIn("access-plain", config["formatters"])
        self.assertIn("access-json", config["formatters"])

        # Verify handlers exist
        self.assertIn("default", config["handlers"])
        self.assertIn("access", config["handlers"])

        # Verify loggers exist
        self.assertIn("uvicorn", config["loggers"])
        self.assertIn("uvicorn.access", config["loggers"])
        self.assertIn("httpx", config["loggers"])
        self.assertIn("httpcore", config["loggers"])
        self.assertIn("marqo_query", config["loggers"])

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_log_formats(self, mock_dict_config):
        """Test that log format selection affects formatter configuration."""
        test_cases = [
            # (log_format, expected_default_formatter, expected_access_formatter)
            (LogFormat.PLAIN, "default-plain", "access-plain"),
            (LogFormat.JSON, "default-json", "access-json"),
        ]

        for log_format, expected_default, expected_access in test_cases:
            with self.subTest(log_format=log_format):
                mock_dict_config.reset_mock()
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_format = log_format
                mock_settings.marqo_log_level = LogLevel.INFO

                instantiate_logger(mock_settings)

                config = mock_dict_config.call_args[0][0]

                # Verify correct formatters are assigned to handlers
                self.assertEqual(
                    expected_default,
                    config["handlers"]["default"]["formatter"],
                )
                self.assertEqual(
                    expected_access,
                    config["handlers"]["access"]["formatter"],
                )

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_log_levels(self, mock_dict_config):
        """Test that log level is applied to all relevant loggers."""
        test_cases = [
            # (log_level, httpx_expected_level)
            (LogLevel.DEBUG, "WARNING"),  # httpx stays WARNING except for ERROR
            (LogLevel.INFO, "WARNING"),
            (LogLevel.WARNING, "WARNING"),
            (LogLevel.ERROR, "ERROR"),  # httpx becomes ERROR when main is ERROR
        ]

        for log_level, httpx_expected in test_cases:
            with self.subTest(log_level=log_level):
                mock_dict_config.reset_mock()
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_format = LogFormat.PLAIN
                mock_settings.marqo_log_level = log_level

                instantiate_logger(mock_settings)

                config = mock_dict_config.call_args[0][0]

                # Verify main log level is set correctly for root and uvicorn
                self.assertEqual(log_level.value, config["root"]["level"])
                self.assertEqual(log_level.value, config["loggers"]["uvicorn"]["level"])

                # uvicorn.access is always INFO
                self.assertEqual("INFO", config["loggers"]["uvicorn.access"]["level"])

                # Verify httpx/httpcore special handling (WARNING except when ERROR)
                self.assertEqual(httpx_expected, config["loggers"]["httpx"]["level"])
                self.assertEqual(httpx_expected, config["loggers"]["httpcore"]["level"])

                # marqo_query always stays WARNING
                self.assertEqual("WARNING", config["loggers"]["marqo_query"]["level"])

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_formatter_details(self, mock_dict_config):
        """Test that formatters are configured with correct details."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        instantiate_logger(mock_settings)

        config = mock_dict_config.call_args[0][0]

        # Check PLAIN formatters have format strings
        self.assertIn("format", config["formatters"]["default-plain"])
        self.assertIn("fmt", config["formatters"]["access-plain"])

        # Check JSON formatters use pythonjsonlogger
        self.assertEqual(
            "pythonjsonlogger.orjson.OrjsonFormatter",
            config["formatters"]["default-json"]["()"],
        )
        self.assertIn("rename_fields", config["formatters"]["default-json"])

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_handler_configuration(self, mock_dict_config):
        """Test that handlers are configured correctly."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        instantiate_logger(mock_settings)

        config = mock_dict_config.call_args[0][0]

        # Check default handler
        self.assertEqual(
            "logging.StreamHandler", config["handlers"]["default"]["class"]
        )
        self.assertEqual("default-plain", config["handlers"]["default"]["formatter"])

        # Check access handler
        self.assertEqual("logging.StreamHandler", config["handlers"]["access"]["class"])
        self.assertEqual("ext://sys.stdout", config["handlers"]["access"]["stream"])
        self.assertEqual("access-plain", config["handlers"]["access"]["formatter"])

    @patch("logging.config.dictConfig")
    def test_instantiate_logger_logger_configuration(self, mock_dict_config):
        """Test that individual loggers are configured correctly."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        instantiate_logger(mock_settings)

        config = mock_dict_config.call_args[0][0]

        # Check uvicorn logger
        self.assertEqual(["default"], config["loggers"]["uvicorn"]["handlers"])
        self.assertFalse(config["loggers"]["uvicorn"]["propagate"])

        # Check uvicorn.access logger
        self.assertEqual(["access"], config["loggers"]["uvicorn.access"]["handlers"])
        self.assertFalse(config["loggers"]["uvicorn.access"]["propagate"])

        # Check httpx logger
        self.assertEqual(["default"], config["loggers"]["httpx"]["handlers"])
        self.assertFalse(config["loggers"]["httpx"]["propagate"])

        # Check marqo_query logger
        self.assertEqual(["default"], config["loggers"]["marqo_query"]["handlers"])
        self.assertFalse(config["loggers"]["marqo_query"]["propagate"])


class TestGetLogger(TestCase):
    """Test class for get_logger function."""

    def test_get_logger_returns_logger_with_correct_name(self):
        """Test that get_logger returns a logger with the correct name."""
        test_cases = [
            "test_logger",
            "test.module.name",
            "logger1",
            "module.logger2",
            "app.services.logger3",
            "__main__",
        ]

        for logger_name in test_cases:
            with self.subTest(logger_name=logger_name):
                logger = get_logger(logger_name)

                self.assertIsInstance(logger, logging.Logger)
                self.assertEqual(logger_name, logger.name)

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that calling get_logger with the same name returns the same instance."""
        logger_name = "test.logger"

        logger1 = get_logger(logger_name)
        logger2 = get_logger(logger_name)

        # Should return the same instance
        self.assertIs(logger1, logger2)

    def test_get_logger_with_empty_string_returns_root_logger(self):
        """Test get_logger with empty string returns root logger."""
        logger = get_logger("")

        # Empty string should return root logger
        self.assertEqual(logging.root, logger)

    @patch("logging.getLogger")
    def test_get_logger_calls_logging_getLogger(self, mock_getLogger):
        """Test that get_logger internally calls logging.getLogger."""
        logger_name = "test.module"
        mock_logger = MagicMock(spec=logging.Logger)
        mock_getLogger.return_value = mock_logger

        result = get_logger(logger_name)

        # Verify logging.getLogger was called with correct name
        mock_getLogger.assert_called_once_with(logger_name)
        self.assertEqual(mock_logger, result)
