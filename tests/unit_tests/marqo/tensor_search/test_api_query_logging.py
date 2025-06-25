import os
import unittest
from unittest.mock import patch, MagicMock

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from marqo.tensor_search.api import app
from marqo.tensor_search.enums import EnvVars
from unit_tests.marqo_test import MarqoTestCase


class TestAPIQueryLogging(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.index_name = "test_index"
        self.search_query = {
            "q": "test query",
            "limit": 10,
            "searchMethod": "TENSOR"
        }
        
        self.mock_tensor_search_patcher = patch('marqo.tensor_search.api.tensor_search.search')
        self.mock_tensor_search = self.mock_tensor_search_patcher.start()
        self.mock_tensor_search.return_value = {"hits": []}

    def tearDown(self):
        self.mock_tensor_search_patcher.stop()

    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_query_logging_enabled_default_env_vars(self, mock_time, mock_marqo_query_logger):
        """Test that slow queries are logged when query details logging is enabled"""

        # this tests the default value for env vars
        # EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        # EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500"

        # the elapsed time is set to 0.5s = 500ms
        mock_time.perf_counter.side_effect = [0.0, 0.5]

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

        # Verify
        self.assertEqual(response.status_code, 200)
        mock_marqo_query_logger.warning.assert_called_once()
        warning_call = mock_marqo_query_logger.warning.call_args[0][0]
        self.assertIn("Slow search query detected: 500.0ms", warning_call)
        self.assertIn("Query:", warning_call)
        self.assertIn("test query", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
    })
    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_query_logging_disabled(self, mock_time, mock_marqo_query_logger):
        """Test that slow queries are not logged when query details logging is disabled"""
        mock_time.perf_counter.side_effect = [0.0, 0.6]  # 600ms

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

        # Verify
        self.assertEqual(response.status_code, 200)
        mock_marqo_query_logger.warning.assert_not_called()  # Should not log when details disabled

    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_fast_query_no_logging(self, mock_time, mock_marqo_query_logger):
        """Test that fast queries are not logged"""
        # Setup
        mock_time.perf_counter.side_effect = [0.0, 0.499]  # 499ms

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

        # Verify
        self.assertEqual(response.status_code, 200)
        mock_marqo_query_logger.warning.assert_not_called()

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1000"
    })
    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_custom_threshold_configuration(self, mock_time, mock_marqo_query_logger):
        """Test that the query uses the configured threshold from environment variables"""
        # Test with time just under threshold - should not log
        mock_time.perf_counter.side_effect = [0.0, 0.999]  # 999ms

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

        # Verify
        self.assertEqual(response.status_code, 200)
        mock_marqo_query_logger.warning.assert_not_called()  # Should not log as it's under the 1000ms threshold

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1000"
    })
    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_exceed_custom_threshold_configuration(self, mock_time, mock_marqo_query_logger):
        """Test that the query uses the configured threshold from environment variables"""
        # Test with time just under threshold - should not log
        mock_time.perf_counter.side_effect = [0.0, 1.0]  # 1000ms

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

        # Verify
        self.assertEqual(response.status_code, 200)
        mock_marqo_query_logger.warning.assert_called_once()
        warning_call = mock_marqo_query_logger.warning.call_args[0][0]
        self.assertIn("Slow search query detected: 1000.0ms", warning_call)
        self.assertIn("Query:", warning_call)
        self.assertIn("test query", warning_call)

    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_search_error_logging_enabled(self, mock_time, mock_marqo_query_logger):
        """Test that search errors are logged with details when logging is enabled"""
        # Setup
        # Mock tensor_search to raise an exception
        self.mock_tensor_search.side_effect = Exception("Search failed")
        mock_time.perf_counter.side_effect = [0.0, 0.3]  # 300ms, fast

        # Execute - allow the client to fail
        try:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
            # If no exception was raised by the client, check the response
            self.assertEqual(response.status_code, 500)
        except:
            # It's ok if the client raises an exception, we're testing the logging
            pass

        # Verify error was logged by our search method with details
        mock_marqo_query_logger.error.assert_called_once()
        error_call = mock_marqo_query_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn("Search failed", error_call)
        self.assertIn("Query:", error_call)

    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.api.parse_request_object')
    @patch('marqo.tensor_search.telemetry.time')
    def test_validation_error_logging_enabled(self, mock_time, mock_parse_request, mock_marqo_query_logger):
        """Test that validation errors are logged with details when logging is enabled"""
        # Mock parse_request_object to raise a validation error
        mock_parse_request.side_effect = RequestValidationError([
            {"loc": ["q"], "msg": "field required", "type": "value_error.missing"}
        ])
        mock_time.perf_counter.side_effect = [0.0, 0.3]

        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json={})

        # Verify error response (should be 422 due to validation error)
        self.assertEqual(response.status_code, 422)

        # Verify error was logged with details
        mock_marqo_query_logger.error.assert_called_once()
        error_call = mock_marqo_query_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn("Query:", error_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
    })
    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_search_error_logging_disabled(self, mock_time, mock_marqo_query_logger):
        """Test that search errors are logged without details when logging is disabled"""
        # Mock tensor_search to raise an exception
        self.mock_tensor_search.side_effect = Exception("Search failed")
        mock_time.perf_counter.side_effect = [0.0, 0.3]

        # Execute - allow the client to fail
        try:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
            # If no exception was raised by the client, check the response
            self.assertEqual(response.status_code, 500)
        except:
            # It's ok if the client raises an exception, we're testing the logging
            pass

        # Verify error was not logged when details disabled
        mock_marqo_query_logger.error.assert_not_called()  # Should not log when details disabled

    @patch('marqo.tensor_search.api.marqo_query_logger')
    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_and_error_query_logs_error_only(self, mock_time, mock_marqo_query_logger):
        """Test that queries that are both slow AND error out only log the error (not slow query)"""
        # Mock tensor_search to raise an exception
        self.mock_tensor_search.side_effect = Exception("Search failed")
        mock_time.perf_counter.side_effect = [0.0, 0.6]

        # Execute - allow the client to fail
        try:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
            # If no exception was raised by the client, check the response
            self.assertEqual(response.status_code, 500)
        except:
            # It's ok if the client raises an exception, we're testing the logging
            pass

        # Verify error was logged but slow query was NOT logged
        mock_marqo_query_logger.error.assert_called_once()
        error_call = mock_marqo_query_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn("Search failed", error_call)
        self.assertIn("Query:", error_call)

        # Verify slow query warning was NOT called (because error was logged first)
        mock_marqo_query_logger.warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()