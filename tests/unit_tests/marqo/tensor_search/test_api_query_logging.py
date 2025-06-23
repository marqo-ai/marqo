import unittest
from unittest.mock import patch, MagicMock

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from marqo.tensor_search.api import app
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.telemetry import RequestMetrics


class TestAPIQueryLogging(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.index_name = "test_index"
        self.search_query = {
            "q": "test query",
            "limit": 10,
            "searchMethod": "TENSOR"
        }
        
        # Mock RequestMetricsStore to avoid context variable issues
        self.mock_request_metrics = MagicMock(spec=RequestMetrics)
        self.mock_request_metrics.time.return_value.__enter__ = MagicMock()
        self.mock_request_metrics.time.return_value.__exit__ = MagicMock(return_value=None)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_slow_query_logging_enabled(self, mock_validate_device, mock_tensor_search, mock_logger, 
                                       mock_read_env, mock_request_store, mock_get_config):
        """Test that slow queries are logged when query details logging is enabled"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "300",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        mock_tensor_search.return_value = {"hits": []}
        
        # Mock the context manager to call the callback with a slow time
        def mock_time_context(key, callback=None):
            if callback:
                callback(600.0)  # Simulate 600ms query (slow)
            return MagicMock()
        
        self.mock_request_metrics.time.side_effect = mock_time_context
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
        
        # Verify
        self.assertEqual(response.status_code, 200)
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Slow search query detected: 600.0ms", warning_call)
        self.assertIn(f"Index: {self.index_name}", warning_call)
        self.assertIn("Query:", warning_call)
        self.assertIn("test query", warning_call)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_slow_query_logging_disabled(self, mock_validate_device, mock_tensor_search, mock_logger, 
                                        mock_read_env, mock_request_store, mock_get_config):
        """Test that slow queries are logged without details when query details logging is disabled"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "300",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        mock_tensor_search.return_value = {"hits": []}
        
        # Mock the context manager to call the callback with a slow time
        def mock_time_context(key, callback=None):
            if callback:
                callback(600.0)  # Simulate 600ms query (slow)
            return MagicMock()
        
        self.mock_request_metrics.time.side_effect = mock_time_context
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
        
        # Verify
        self.assertEqual(response.status_code, 200)
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Slow search query detected: 600.0ms", warning_call)
        self.assertIn(f"Index: {self.index_name}", warning_call)
        self.assertNotIn("Query:", warning_call)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_fast_query_no_logging(self, mock_validate_device, mock_tensor_search, mock_logger, 
                                   mock_read_env, mock_request_store, mock_get_config):
        """Test that fast queries are not logged"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        mock_tensor_search.return_value = {"hits": []}
        
        # Mock the context manager to call the callback with a fast time
        def mock_time_context(key, callback=None):
            if callback:
                callback(200.0)  # Simulate 200ms query (fast)
            return MagicMock()
        
        self.mock_request_metrics.time.side_effect = mock_time_context
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
        
        # Verify
        self.assertEqual(response.status_code, 200)
        mock_logger.warning.assert_not_called()

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')  # Mock the search method logger
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_search_error_logging_enabled(self, mock_validate_device, mock_tensor_search, mock_search_logger, 
                                         mock_read_env, mock_request_store, mock_get_config):
        """Test that search errors are logged with details when logging is enabled"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        self.mock_request_metrics.time.return_value = MagicMock()
        
        # Mock tensor_search to raise an exception
        mock_tensor_search.side_effect = Exception("Search failed")
        
        # Execute - allow the client to fail
        try:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
            # If no exception was raised by the client, check the response
            self.assertEqual(response.status_code, 500)
        except:
            # It's ok if the client raises an exception, we're testing the logging
            pass
        
        # Verify error was logged by our search method with details
        mock_search_logger.error.assert_called_once()
        error_call = mock_search_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn(f"Index: {self.index_name}", error_call)
        self.assertIn("Search failed", error_call)
        self.assertIn("Query:", error_call)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')
    @patch('marqo.tensor_search.api.parse_request_object')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_validation_error_logging_enabled(self, mock_validate_device, mock_parse_request, mock_logger, 
                                             mock_read_env, mock_request_store, mock_get_config):
        """Test that validation errors are logged with details when logging is enabled"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        self.mock_request_metrics.time.return_value = MagicMock()
        
        # Mock parse_request_object to raise a validation error
        mock_parse_request.side_effect = RequestValidationError([
            {"loc": ["q"], "msg": "field required", "type": "value_error.missing"}
        ])
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json={})
        
        # Verify error response (should be 422 due to validation error)
        self.assertEqual(response.status_code, 422)
        
        # Verify error was logged with details
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn(f"Index: {self.index_name}", error_call)
        self.assertIn("Query:", error_call)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.logger')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_search_error_logging_disabled(self, mock_validate_device, mock_tensor_search, mock_logger, 
                                          mock_read_env, mock_request_store, mock_get_config):
        """Test that search errors are logged without details when logging is disabled"""
        # Setup
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        self.mock_request_metrics.time.return_value = MagicMock()
        
        # Mock tensor_search to raise an exception
        mock_tensor_search.side_effect = Exception("Search failed")
        
        # Execute - allow the client to fail
        try:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
            # If no exception was raised by the client, check the response
            self.assertEqual(response.status_code, 500)
        except:
            # It's ok if the client raises an exception, we're testing the logging
            pass
        
        # Verify error was logged without details
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("Failed search query", error_call)
        self.assertIn(f"Index: {self.index_name}", error_call)
        self.assertIn("Search failed", error_call)
        self.assertNotIn("Query:", error_call)

    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.RequestMetricsStore.for_request')
    @patch('marqo.tensor_search.api.utils.read_env_vars_and_defaults')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_custom_threshold_configuration(self, mock_validate_device, mock_tensor_search, 
                                           mock_read_env, mock_request_store, mock_get_config):
        """Test that the query uses the configured threshold from environment variables"""
        # Setup with custom threshold
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        mock_read_env.side_effect = lambda env_var: {
            EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1000",  # Custom threshold
            EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
        }.get(env_var, "500")
        
        mock_request_store.return_value = self.mock_request_metrics
        mock_tensor_search.return_value = {"hits": []}
        
        callback_called_with = []
        
        def mock_time_context(key, callback=None):
            if callback:
                # Test with time just under threshold - should not log
                callback(999.0)
                callback_called_with.append(999.0)
            return MagicMock()
        
        self.mock_request_metrics.time.side_effect = mock_time_context
        
        # Execute
        with patch('marqo.tensor_search.api.logger') as mock_logger:
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)
        
        # Verify
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(callback_called_with), 1)
        mock_logger.warning.assert_not_called()  # Should not log as it's under the 1000ms threshold


if __name__ == '__main__':
    unittest.main()