import importlib
import io
import logging
import os
import unittest
from unittest.mock import patch

from marqo import logging as marqo_logging
from marqo.api.exceptions import EnvVarError
from marqo.logging import LogLevel, LogFormat


class TestLoggingConfig(unittest.TestCase):
    def setUp(self):
        # Clear any existing logging configuration
        marqo_logging.get_logger().handlers = []

    def tearDown(self):
        # Reset logging after each test
        marqo_logging.get_logger().handlers = []

    def test_valid_log_levels(self):
        """Test that only valid log levels are accepted"""
        # Test each valid level
        for level in LogLevel:
            with patch.dict(os.environ, {'MARQO_LOG_LEVEL': level, 'MARQO_LOG_FORMAT': 'plain'}):
                # Should not raise an exception
                importlib.reload(marqo_logging)  # Need to reload module to re-execute the config

        # Test invalid level
        with patch.dict(os.environ, {'MARQO_LOG_LEVEL': 'invalid', 'MARQO_LOG_FORMAT': 'plain'}):
            with self.assertRaises(EnvVarError) as context:
                importlib.reload(marqo_logging)
            self.assertIn('`MARQO_LOG_LEVEL` = `invalid` is not supported', str(context.exception))

    def test_valid_log_formats(self):
        """Test that only valid log formats are accepted"""
        # Test each valid format
        for fmt in LogFormat:
            with patch.dict(os.environ, {'MARQO_LOG_FORMAT': fmt, 'MARQO_LOG_LEVEL': 'info'}):
                # Should not raise an exception
                importlib.reload(marqo_logging)

        # Test invalid format
        with patch.dict(os.environ, {'MARQO_LOG_FORMAT': 'invalid', 'MARQO_LOG_LEVEL': 'info'}):
            with self.assertRaises(EnvVarError) as context:
                importlib.reload(marqo_logging)
            self.assertIn('`MARQO_LOG_FORMAT` = `invalid` is not supported', str(context.exception))

    def test_config_structure_plain(self):
        """Test the configuration structure with plain format"""
        with patch.dict(os.environ, {'MARQO_LOG_FORMAT': 'plain', 'MARQO_LOG_LEVEL': 'debug'}):
            importlib.reload(marqo_logging)
            from marqo.logging import LOGGING_CONFIG

            self.assertEqual(LOGGING_CONFIG['handlers']['default']['formatter'], 'default-plain')
            self.assertEqual(LOGGING_CONFIG['handlers']['access']['formatter'], 'access-plain')

    def test_config_structure_json(self):
        """Test the configuration structure with json format"""
        with patch.dict(os.environ, {'MARQO_LOG_FORMAT': 'json', 'MARQO_LOG_LEVEL': 'debug'}):
            importlib.reload(marqo_logging)
            from marqo.logging import LOGGING_CONFIG

            self.assertEqual(LOGGING_CONFIG['handlers']['default']['formatter'], 'default-json')
            self.assertEqual(LOGGING_CONFIG['handlers']['access']['formatter'], 'access-json')

    def test_log_level_propagation(self):
        """Test that log levels are correctly set in the config"""
        test_cases = [
            ('debug', 'DEBUG'),
            ('info', 'INFO'),
            ('warning', 'WARNING'),
            ('error', 'ERROR')
        ]

        for input_level, expected_level in test_cases:
            with patch.dict(os.environ, {'MARQO_LOG_LEVEL': input_level, 'MARQO_LOG_FORMAT': 'plain'}):
                importlib.reload(marqo_logging)
                from marqo.logging import LOGGING_CONFIG

                self.assertEqual(LOGGING_CONFIG['root']['level'], expected_level)
                self.assertEqual(LOGGING_CONFIG['loggers']['uvicorn']['level'], expected_level)

    def test_httpx_log_level(self):
        """Test that httpx logger has special level handling"""
        test_cases = [
            ('error', 'ERROR'),
            ('warning', 'WARNING'),
            ('info', 'WARNING'),
            ('debug', 'WARNING')
        ]

        for input_level, expected_level in test_cases:
            with patch.dict(os.environ, {'MARQO_LOG_LEVEL': input_level, 'MARQO_LOG_FORMAT': 'plain'}):
                importlib.reload(marqo_logging)
                from marqo.logging import LOGGING_CONFIG

                self.assertEqual(LOGGING_CONFIG['loggers']['httpx']['level'], expected_level)

    def test_access_log_level_fixed(self):
        """Test that access log level is always INFO"""
        for level in ['debug', 'info', 'warning', 'error']:
            with patch.dict(os.environ, {'MARQO_LOG_LEVEL': level, 'MARQO_LOG_FORMAT': 'plain'}):
                importlib.reload(marqo_logging)
                from marqo.logging import LOGGING_CONFIG

                self.assertEqual(LOGGING_CONFIG['loggers']['uvicorn.access']['level'], 'INFO')

    def test_plain_format_output(self):
        """Test the exact plain text format output"""
        # Set up environment for plain format
        with patch.dict('os.environ', {
            'MARQO_LOG_LEVEL': 'info',
            'MARQO_LOG_FORMAT': 'plain'
        }):
            importlib.reload(marqo_logging)
            # replace the stream object of the handler
            stream = io.StringIO()
            logging.root.handlers[0].stream = stream

            logger = logging.getLogger('test')
            logger.info("Test message")

            # Verify output format
            output = stream.getvalue()
            self.assertRegex(output,
                r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \d+ INFO test: Test message\n$')

    def test_json_format_output(self):
        """Test the exact JSON format output"""
        # Set up environment for JSON format
        with patch.dict('os.environ', {
            'MARQO_LOG_LEVEL': 'info',
            'MARQO_LOG_FORMAT': 'json'
        }):
            importlib.reload(marqo_logging)

            # replace the stream object of the handler
            stream = io.StringIO()
            logging.root.handlers[0].stream = stream

            # Generate log message
            logger = marqo_logging.get_logger('test')
            logger.info("Test message")

            # Verify JSON output
            output = stream.getvalue()
            import json
            try:
                log_data = json.loads(output)
                self.assertEqual(log_data['level'], 'INFO')
                self.assertEqual(log_data['name'], 'test')
                self.assertEqual(log_data['message'], 'Test message')
                self.assertIn('timestamp', log_data)
                self.assertIn('process', log_data)
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")

    def test_access_log_plain_format(self):
        """Test the exact plain text format of access logs"""
        # Set up environment for plain format
        with patch.dict('os.environ', {
            'MARQO_LOG_LEVEL': 'info',
            'MARQO_LOG_FORMAT': 'plain'
        }):
            importlib.reload(marqo_logging)

            # Get the access logger and its handler
            access_logger = logging.getLogger('uvicorn.access')
            access_handler = access_logger.handlers[0]

            # Replace the stream object of the handler
            stream = io.StringIO()
            access_handler.stream = stream

            access_logger.info('%s - "%s %s HTTP/%s" %d',
                               "127.0.0.1:30000", "GET", "/?debug=true", "1.1", 200)

            # Verify output format
            output = stream.getvalue()
            self.assertRegex(output,
                             r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \d+ '
                             r'INFO 127.0.0.1:30000 - "GET /\?debug=true HTTP/1.1" 200 OK\n$')

    def test_access_log_json_format(self):
        """Test the exact JSON format of access logs"""
        # Set up environment for JSON format
        with patch.dict('os.environ', {
            'MARQO_LOG_LEVEL': 'info',
            'MARQO_LOG_FORMAT': 'json'
        }):
            importlib.reload(marqo_logging)

            # Get the access logger and its handler
            access_logger = logging.getLogger('uvicorn.access')
            access_handler = access_logger.handlers[0]

            # Replace the stream object of the handler
            stream = io.StringIO()
            access_handler.stream = stream

            access_logger.info('%s - "%s %s HTTP/%s" %d',
                               "127.0.0.1:30000", "GET", "/api/v1/search", "1.1", 200)

            # Verify JSON output
            output = stream.getvalue()
            import json
            try:
                log_data = json.loads(output)
                self.assertEqual(log_data['level'], 'INFO')
                self.assertEqual(log_data['client_addr'], '127.0.0.1:30000')
                self.assertEqual(log_data['request_line'], 'GET /api/v1/search HTTP/1.1')
                self.assertEqual(log_data['status_code'], "200 OK")
                self.assertIn('timestamp', log_data)
                self.assertIn('process', log_data)
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")

    def test_exception_traceback_is_logged_as_string_array_with_json_formatter(self):
        """Test the exact JSON format of access logs"""
        # Set up environment for JSON format
        with patch.dict('os.environ', {
            'MARQO_LOG_LEVEL': 'info',
            'MARQO_LOG_FORMAT': 'json'
        }):
            importlib.reload(marqo_logging)

            # replace the stream object of the handler
            stream = io.StringIO()
            logging.root.handlers[0].stream = stream

            # Generate log message
            logger = marqo_logging.get_logger('test')
            try:
                raise ValueError('test exception')
            except ValueError as e:
                logger.error("OOPS, error occurred", exc_info=True)

            output = stream.getvalue()
            import json
            try:
                log_data = json.loads(output)
                self.assertEqual(log_data['level'], 'ERROR')
                self.assertEqual(log_data['name'], 'test')
                self.assertEqual(log_data['message'], 'OOPS, error occurred')
                self.assertIn('timestamp', log_data)
                self.assertIn('process', log_data)
                self.assertIsInstance(log_data['exc_info'], list)
                self.assertEqual('Traceback (most recent call last):', log_data['exc_info'][0])
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")
