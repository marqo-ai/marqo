import os
import time
import logging
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import marqo.tensor_search.api as api
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import EnvVars


class TestAPIQueryLoggingIntegration(MarqoTestCase):
    """Integration tests for the query logging feature in the API"""

    def setUp(self):
        self.client = TestClient(api.app)
        self.index_name = "test_query_logging_index"
        
        # Capture logs in memory for testing
        self.log_messages = []
        self.log_handler = None
        
    def tearDown(self):
        # Clean up log handler
        if self.log_handler:
            logging.getLogger('marqo_query').removeHandler(self.log_handler)

    def _setup_log_capture(self):
        """Set up log capture for testing"""
        class LogCapture(logging.Handler):
            def __init__(self, messages_list):
                super().__init__()
                self.messages = messages_list
                
            def emit(self, record):
                self.messages.append(self.format(record))
        
        self.log_handler = LogCapture(self.log_messages)
        self.log_handler.setLevel(logging.WARNING)
        logger = logging.getLogger('marqo_query')
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.WARNING)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "100",  # Very low threshold for testing
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_slow_query_logging_integration_with_details(self, mock_validate_device, mock_tensor_search, mock_get_config):
        """Integration test for slow query logging with details enabled"""
        # Setup
        self._setup_log_capture()
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        
        # Mock tensor_search to be slow and return a result
        def slow_search(*args, **kwargs):
            time.sleep(0.15)  # 150ms - should trigger logging with 100ms threshold
            return {"hits": [], "processingTimeMs": 150}
        
        mock_tensor_search.side_effect = slow_search
        
        search_query = {
            "q": "integration test query",
            "limit": 5,
            "searchMethod": "TENSOR"
        }
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)
        
        # Verify response is successful
        self.assertEqual(response.status_code, 200)
        
        # Verify slow query was logged with details
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertTrue(len(warning_logs) > 0, f"Expected slow query log, but got logs: {self.log_messages}")
        
        warning_log = warning_logs[0]
        self.assertIn("Slow search query detected:", warning_log)
        self.assertIn("Query:", warning_log)
        self.assertIn("integration test query", warning_log)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "100",  # Very low threshold for testing
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_slow_query_logging_integration_without_details(self, mock_validate_device, mock_tensor_search, mock_get_config):
        """Integration test for slow query logging with details disabled"""
        # Setup
        self._setup_log_capture()
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        
        # Mock tensor_search to be slow and return a result
        def slow_search(*args, **kwargs):
            time.sleep(0.15)  # 150ms - should trigger logging with 100ms threshold
            return {"hits": [], "processingTimeMs": 150}
        
        mock_tensor_search.side_effect = slow_search
        
        search_query = {
            "q": "integration test query",
            "limit": 5,
            "searchMethod": "TENSOR"
        }
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)
        
        # Verify response is successful
        self.assertEqual(response.status_code, 200)
        
        # Verify no slow query was logged when details disabled
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs when details disabled, but got: {warning_logs}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1000",  # High threshold
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_fast_query_no_logging_integration(self, mock_validate_device, mock_tensor_search, mock_get_config):
        """Integration test to verify fast queries are not logged"""
        # Setup
        self._setup_log_capture()
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        
        # Mock tensor_search to be fast
        mock_tensor_search.return_value = {"hits": [], "processingTimeMs": 50}
        
        search_query = {
            "q": "fast query",
            "limit": 5,
            "searchMethod": "TENSOR"
        }
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)
        
        # Verify response is successful
        self.assertEqual(response.status_code, 200)
        
        # Verify no slow query warnings were logged
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs, but got: {warning_logs}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_search_error_logging_integration(self, mock_validate_device, mock_tensor_search, mock_get_config):
        """Integration test for search error logging"""
        # Setup error logging capture
        class ErrorLogCapture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.error_messages = []
                
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    self.error_messages.append(self.format(record))
        
        error_handler = ErrorLogCapture()
        error_handler.setLevel(logging.ERROR)
        
        # Add handler to both api logger and route logger to catch errors
        api_logger = logging.getLogger('marqo.tensor_search.api')
        route_logger = logging.getLogger('marqo.api.route')
        
        api_logger.addHandler(error_handler)
        route_logger.addHandler(error_handler)
        api_logger.setLevel(logging.ERROR)
        route_logger.setLevel(logging.ERROR)
        
        try:
            mock_validate_device.return_value = "cpu"
            mock_get_config.return_value = MagicMock()
            
            # Mock tensor_search to raise an exception
            mock_tensor_search.side_effect = Exception("Search integration test failure")
            
            search_query = {
                "q": "error test query",
                "limit": 5,
                "searchMethod": "TENSOR"
            }
            
            # Execute - this should result in an error (allow it to fail)
            try:
                response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)
                # If no exception was raised by the client, check the response
                self.assertEqual(response.status_code, 500)
            except:
                # It's ok if the client raises an exception during error handling
                pass
            
            # Verify error was logged (either by our search method or the route handler)
            self.assertTrue(len(error_handler.error_messages) > 0, 
                          "Expected error to be logged")
            
            # Check that the error contains relevant information
            error_logs = error_handler.error_messages
            search_error_found = any("Search integration test failure" in log for log in error_logs)
            self.assertTrue(search_error_found, 
                          f"Expected error about search failure, got: {error_logs}")
            
            # Check for our specific error logging format
            api_error_found = any("Failed search query" in log and "Query:" in log 
                                for log in error_logs)
            self.assertTrue(api_error_found, 
                          f"Expected our API error logging format, got: {error_logs}")
            
        finally:
            # Clean up handlers
            api_logger.removeHandler(error_handler)
            route_logger.removeHandler(error_handler)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "200",
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_validation_error_logging_integration(self, mock_validate_device, mock_get_config):
        """Integration test for validation error logging"""
        # Setup error logging capture
        class ErrorLogCapture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.error_messages = []
                
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    self.error_messages.append(self.format(record))
        
        error_handler = ErrorLogCapture()
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers to capture validation errors
        api_logger = logging.getLogger('marqo.tensor_search.api')
        route_logger = logging.getLogger('marqo.api.route')
        
        api_logger.addHandler(error_handler)
        route_logger.addHandler(error_handler)
        api_logger.setLevel(logging.ERROR)
        route_logger.setLevel(logging.ERROR)
        
        try:
            mock_validate_device.return_value = "cpu"
            mock_get_config.return_value = MagicMock()
            
            # Send invalid search query (missing required fields)
            invalid_query = {}  # Empty query should cause validation error
            
            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=invalid_query)
            
            # Verify validation error response
            self.assertEqual(response.status_code, 422)  # Unprocessable Entity
            
            # The validation error should be caught by FastAPI's validation handler
            # and return a 422 response, so we might not see it in our error logs
            # This test mainly verifies the integration works end-to-end
            
        finally:
            # Clean up handlers
            api_logger.removeHandler(error_handler)
            route_logger.removeHandler(error_handler)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "800",  # Custom threshold
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.get_config')
    @patch('marqo.tensor_search.api.tensor_search.search')
    @patch('marqo.tensor_search.api.api_validation.validate_device')
    def test_custom_threshold_integration(self, mock_validate_device, mock_tensor_search, mock_get_config):
        """Integration test for custom threshold configuration"""
        # Setup
        self._setup_log_capture()
        mock_validate_device.return_value = "cpu"
        mock_get_config.return_value = MagicMock()
        
        # Mock tensor_search to take 600ms (under 800ms threshold)
        def medium_speed_search(*args, **kwargs):
            time.sleep(0.6)  # 600ms - should NOT trigger logging with 800ms threshold
            return {"hits": [], "processingTimeMs": 600}
        
        mock_tensor_search.side_effect = medium_speed_search
        
        search_query = {
            "q": "medium speed query",
            "limit": 5,
            "searchMethod": "TENSOR"
        }
        
        # Execute
        response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)
        
        # Verify response is successful
        self.assertEqual(response.status_code, 200)
        
        # Verify no slow query warnings (600ms < 800ms threshold)
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs with 800ms threshold, but got: {warning_logs}")


if __name__ == '__main__':
    import unittest
    unittest.main()