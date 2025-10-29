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

    def test_instantiate_logger_with_plain_format_info_level(self):
        """Test logger instantiation with PLAIN format and INFO level."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            # Verify dictConfig was called
            mock_dict_config.assert_called_once()

            # Get the config that was passed
            config = mock_dict_config.call_args[0][0]

            # Verify format is PLAIN
            self.assertIn("default-plain", config["formatters"])
            self.assertEqual(
                "default-plain", config["handlers"]["default"]["formatter"]
            )

            # Verify log level is INFO
            self.assertEqual("INFO", config["root"]["level"])
            self.assertEqual("INFO", config["loggers"]["uvicorn"]["level"])

    def test_instantiate_logger_with_json_format_debug_level(self):
        """Test logger instantiation with JSON format and DEBUG level."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.JSON
        mock_settings.marqo_log_level = LogLevel.DEBUG

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            # Verify format is JSON
            self.assertIn("default-json", config["formatters"])
            self.assertEqual("default-json", config["handlers"]["default"]["formatter"])

            # Verify log level is DEBUG
            self.assertEqual("DEBUG", config["root"]["level"])
            self.assertEqual("DEBUG", config["loggers"]["uvicorn"]["level"])

    def test_instantiate_logger_with_warning_level(self):
        """Test logger instantiation with WARNING level."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.WARNING

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Verify log level is WARNING
            self.assertEqual("WARNING", config["root"]["level"])
            self.assertEqual("WARNING", config["loggers"]["uvicorn"]["level"])
            self.assertEqual("WARNING", config["loggers"]["httpx"]["level"])
            self.assertEqual("WARNING", config["loggers"]["httpcore"]["level"])

    def test_instantiate_logger_with_error_level(self):
        """Test logger instantiation with ERROR level."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.ERROR

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Verify log level is ERROR
            self.assertEqual("ERROR", config["root"]["level"])
            self.assertEqual("ERROR", config["loggers"]["uvicorn"]["level"])
            self.assertEqual("ERROR", config["loggers"]["httpx"]["level"])
            self.assertEqual("ERROR", config["loggers"]["httpcore"]["level"])

    def test_instantiate_logger_all_log_formats(self):
        """Test logger instantiation with all log format options."""
        test_cases = [
            (LogFormat.PLAIN, "default-plain", "access-plain"),
            (LogFormat.JSON, "default-json", "access-json"),
        ]

        for (
            log_format,
            expected_default_formatter,
            expected_access_formatter,
        ) in test_cases:
            with self.subTest(log_format=log_format):
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_format = log_format
                mock_settings.marqo_log_level = LogLevel.INFO

                with patch("logging.config.dictConfig") as mock_dict_config:
                    instantiate_logger(mock_settings)

                    config = mock_dict_config.call_args[0][0]

                    # Verify formatters are configured correctly
                    self.assertEqual(
                        expected_default_formatter,
                        config["handlers"]["default"]["formatter"],
                    )
                    self.assertEqual(
                        expected_access_formatter,
                        config["handlers"]["access"]["formatter"],
                    )

    def test_instantiate_logger_all_log_levels(self):
        """Test logger instantiation with all log level options."""
        test_cases = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
        ]

        for log_level in test_cases:
            with self.subTest(log_level=log_level):
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_format = LogFormat.PLAIN
                mock_settings.marqo_log_level = log_level

                with patch("logging.config.dictConfig") as mock_dict_config:
                    instantiate_logger(mock_settings)

                    config = mock_dict_config.call_args[0][0]

                    # Verify log level is set correctly
                    self.assertEqual(log_level.value, config["root"]["level"])

    def test_instantiate_logger_configuration_structure(self):
        """Test that logger configuration has the correct structure."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Verify basic structure
            self.assertEqual(1, config["version"])
            self.assertFalse(config["disable_existing_loggers"])

            # Verify formatters
            self.assertIn("formatters", config)
            self.assertIn("default-plain", config["formatters"])
            self.assertIn("default-json", config["formatters"])
            self.assertIn("access-plain", config["formatters"])
            self.assertIn("access-json", config["formatters"])

            # Verify handlers
            self.assertIn("handlers", config)
            self.assertIn("default", config["handlers"])
            self.assertIn("access", config["handlers"])

            # Verify loggers
            self.assertIn("loggers", config)
            self.assertIn("uvicorn", config["loggers"])
            self.assertIn("uvicorn.access", config["loggers"])
            self.assertIn("httpx", config["loggers"])
            self.assertIn("httpcore", config["loggers"])
            self.assertIn("marqo_query", config["loggers"])

            # Verify root logger
            self.assertIn("root", config)
            self.assertIn("handlers", config["root"])
            self.assertIn("level", config["root"])

    def test_instantiate_logger_formatters_configuration(self):
        """Test that formatters are configured correctly."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Check plain formatters
            self.assertIn("format", config["formatters"]["default-plain"])
            self.assertIn("fmt", config["formatters"]["access-plain"])

            # Check JSON formatters
            self.assertEqual(
                "pythonjsonlogger.orjson.OrjsonFormatter",
                config["formatters"]["default-json"]["()"],
            )
            self.assertIn("rename_fields", config["formatters"]["default-json"])

    def test_instantiate_logger_handlers_configuration(self):
        """Test that handlers are configured correctly."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Check default handler
            self.assertEqual(
                "logging.StreamHandler", config["handlers"]["default"]["class"]
            )
            self.assertEqual(
                "default-plain", config["handlers"]["default"]["formatter"]
            )

            # Check access handler
            self.assertEqual(
                "logging.StreamHandler", config["handlers"]["access"]["class"]
            )
            self.assertEqual("ext://sys.stdout", config["handlers"]["access"]["stream"])
            self.assertEqual("access-plain", config["handlers"]["access"]["formatter"])

    def test_instantiate_logger_specific_loggers_configuration(self):
        """Test that specific loggers are configured correctly."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.INFO

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # Check uvicorn logger
            self.assertEqual(["default"], config["loggers"]["uvicorn"]["handlers"])
            self.assertEqual("INFO", config["loggers"]["uvicorn"]["level"])
            self.assertFalse(config["loggers"]["uvicorn"]["propagate"])

            # Check uvicorn.access logger
            self.assertEqual(
                ["access"], config["loggers"]["uvicorn.access"]["handlers"]
            )
            self.assertEqual("INFO", config["loggers"]["uvicorn.access"]["level"])
            self.assertFalse(config["loggers"]["uvicorn.access"]["propagate"])

            # Check httpx logger
            self.assertEqual(["default"], config["loggers"]["httpx"]["handlers"])
            self.assertEqual("WARNING", config["loggers"]["httpx"]["level"])
            self.assertFalse(config["loggers"]["httpx"]["propagate"])

            # Check marqo_query logger
            self.assertEqual(["default"], config["loggers"]["marqo_query"]["handlers"])
            self.assertEqual("WARNING", config["loggers"]["marqo_query"]["level"])
            self.assertFalse(config["loggers"]["marqo_query"]["propagate"])

    def test_instantiate_logger_httpx_level_with_error(self):
        """Test that httpx logger level is set to ERROR when main level is ERROR."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.marqo_log_format = LogFormat.PLAIN
        mock_settings.marqo_log_level = LogLevel.ERROR

        with patch("logging.config.dictConfig") as mock_dict_config:
            instantiate_logger(mock_settings)

            config = mock_dict_config.call_args[0][0]

            # httpx and httpcore should be ERROR when main level is ERROR
            self.assertEqual("ERROR", config["loggers"]["httpx"]["level"])
            self.assertEqual("ERROR", config["loggers"]["httpcore"]["level"])

    def test_instantiate_logger_httpx_level_with_non_error(self):
        """Test that httpx logger level is WARNING when main level is not ERROR."""
        test_cases = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING]

        for log_level in test_cases:
            with self.subTest(log_level=log_level):
                mock_settings = MagicMock(spec=Settings)
                mock_settings.marqo_log_format = LogFormat.PLAIN
                mock_settings.marqo_log_level = log_level

                with patch("logging.config.dictConfig") as mock_dict_config:
                    instantiate_logger(mock_settings)

                    config = mock_dict_config.call_args[0][0]

                    # httpx and httpcore should be WARNING
                    self.assertEqual("WARNING", config["loggers"]["httpx"]["level"])
                    self.assertEqual("WARNING", config["loggers"]["httpcore"]["level"])


class TestGetLogger(TestCase):
    """Test class for get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a logging.Logger instance."""
        logger = get_logger("test_logger")

        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_returns_correct_name(self):
        """Test that get_logger returns logger with the correct name."""
        logger_name = "test.module.name"
        logger = get_logger(logger_name)

        self.assertEqual(logger_name, logger.name)

    def test_get_logger_different_names(self):
        """Test that get_logger returns different loggers for different names."""
        test_cases = [
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

    def test_get_logger_with_empty_string(self):
        """Test get_logger with empty string returns root logger."""
        logger = get_logger("")

        # Empty string should return root logger
        self.assertEqual(logging.root, logger)

    def test_get_logger_calls_logging_getLogger(self):
        """Test that get_logger internally calls logging.getLogger."""
        logger_name = "test.module"

        with patch("logging.getLogger") as mock_getLogger:
            mock_logger = MagicMock(spec=logging.Logger)
            mock_getLogger.return_value = mock_logger

            result = get_logger(logger_name)

            # Verify logging.getLogger was called with correct name
            mock_getLogger.assert_called_once_with(logger_name)
            self.assertEqual(mock_logger, result)
