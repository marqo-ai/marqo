import importlib
import os
import sys
import unittest
from unittest.mock import patch

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search.api import app
from marqo.tensor_search.enums import EnvVars
from tests.unit_tests.marqo_test import MarqoTestCase


class TestAPIQueryLogging(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.index_name = "test_index"
        self.search_query = {
            "q": "test query",
            "searchMethod": "TENSOR",
            "limit": 10
        }
        
        self.mock_tensor_search_patcher = patch('marqo.tensor_search.api.tensor_search.search')
        self.mock_tensor_search = self.mock_tensor_search_patcher.start()
        self.mock_tensor_search.return_value = {"hits": []}

    def tearDown(self):
        self.mock_tensor_search_patcher.stop()

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_query_logging_enabled_default_env_vars(self, mock_time):
        """Test that slow queries are logged when query details logging is enabled"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            # this tests the default value for env vars
            # EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "900"
            # the elapsed time is set to 0.901s = 901ms
            mock_time.perf_counter.side_effect = [0.0, 0.901]

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 901.0ms", warning_call)
            self.assertIn(f"{self.search_query}", warning_call)

    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_query_logging_disabled(self, mock_time):
        """Test that slow queries are not logged when query details logging is disabled"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:

            # this tests the default value for env vars
            # EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
            mock_time.perf_counter.side_effect = [0.0, 1.0]  # 1000ms

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_not_called()  # Should not log when details disabled

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_fast_query_no_logging(self, mock_time):
        """Test that fast queries are not logged"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 0.9]  # 900ms

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_not_called()

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1000"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_custom_threshold_configuration(self, mock_time):
        """Test that the query uses the configured threshold from environment variables"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            # Test with time just under threshold - should not log
            mock_time.perf_counter.side_effect = [0.0, 1.0]  # 1000ms

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_not_called()  # Should not log as it's under the 1000ms threshold

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1000"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_exceed_custom_threshold_configuration(self, mock_time):
        """Test that the query uses the configured threshold from environment variables"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            # Test with time just under threshold - should not log
            mock_time.perf_counter.side_effect = [0.0, 1.001]  # 1001ms

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=self.search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1001.0ms", warning_call)
            self.assertIn(f"{self.search_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_search_error_logging_enabled(self, mock_time):
        """Test that search errors are logged with details when logging is enabled"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
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

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.api.parse_request_object')
    @patch('marqo.tensor_search.telemetry.time')
    def test_validation_error_when_parsing_request_body_are_not_logged(self, mock_time, mock_parse_request):
        """Test that validation errors are logged with details when logging is enabled"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            # Mock parse_request_object to raise a validation error
            mock_parse_request.side_effect = RequestValidationError([
                {"loc": ["q"], "msg": "field required", "type": "value_error.missing"}
            ])
            mock_time.perf_counter.side_effect = [0.0, 0.3]

            # Execute
            search_query = {"limit": 1}
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify error response (should be 422 due to validation error)
            self.assertEqual(response.status_code, 422)
            mock_marqo_query_logger.warning.assert_not_called()

    @patch('marqo.tensor_search.telemetry.time')
    def test_search_error_logging_disabled(self, mock_time):
        """Test that search errors are logged without details when logging is disabled"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
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

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_slow_and_error_query_logs_error_only(self, mock_time):
        """Test that queries that are both slow AND error out only log the error (not slow query)"""
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            # Mock tensor_search to raise an exception
            self.mock_tensor_search.side_effect = Exception("Search failed")
            mock_time.perf_counter.side_effect = [0.0, 1.0]

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
            self.assertIn(f"{self.search_query}", error_call)

            # Verify slow query warning was NOT called (because error was logged first)
            mock_marqo_query_logger.warning.assert_not_called()

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_do_not_log_vectors(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": {
                    "customVector": {
                        "content": "abc",
                        "vector": [0.1] * 768
                    }
                },
                "searchMethod": "TENSOR",
                "limit": 10,
                "context": {
                    "tensor": [
                        {"vector": [0.2] * 768, "weight": 0.2},
                        {"vector": [0.3] * 768, "weight": 0.8},
                    ],
                    # TODO add document ids when PR 1254 is merged
                }
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": {
                    "customVector": {
                        "content": "abc",
                        "vector": []
                    }
                },
                "searchMethod": "TENSOR",
                "limit": 10,
                "context": {
                    "tensor": [
                        {"vector": [], "weight": 0.2},
                        {"vector": [], "weight": 0.8},
                    ]
                }
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_truncate_long_query(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": "this is a long query with more than 20 characters",
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": "this is a long query...[truncated:20/49]",
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_truncate_long_query_in_hybrid_parameter(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "hybridParameters": {
                    "queryLexical": "this is a long lexical query with more than 20 characters",
                    "queryTensor": "this is a long tensor query with more than 20 characters",
                }
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "hybridParameters": {
                    "queryLexical": "this is a long lexic...[truncated:20/57]",
                    "queryTensor": "this is a long tenso...[truncated:20/56]",
                }
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_truncate_long_query_in_dict(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": {
                    "this is a long query with more than 20 characters": 0.3,
                    "this is a short one": 0.2,
                    "and this is another long one": 0.5
                },
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": {
                    "this is a long query...[truncated:20/49]": 0.3,
                    "this is a short one": 0.2,
                    "and this is another ...[truncated:20/28]": 0.5
                },
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_truncate_long_query_in_dict_in_hybrid_parameter(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "hybridParameters": {
                    "queryLexical": "short lexical",
                    "queryTensor": {
                        "this is a long query with more than 20 characters": 0.3,
                        "this is a short one": 0.2,
                        "and this is another long one": 0.5
                    },
                }
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "hybridParameters": {
                    "queryLexical": "short lexical",
                    "queryTensor": {
                        "this is a long query...[truncated:20/49]": 0.3,
                        "this is a short one": 0.2,
                        "and this is another ...[truncated:20/28]": 0.5
                    },
                }
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_truncate_long_custom_vector_query_content(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": {
                    "customVector": {
                        "content": "this is a long query with more than 20 characters",
                        "vector": [0.1] * 768
                    }
                },
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": {
                    "customVector": {
                        "content": "this is a long query...[truncated:20/49]",
                        "vector": []
                    }
                },
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_skip_fields_with_secrets(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": "do not put secrets",
                "searchMethod": "TENSOR",
                "limit": 10,
                "mediaDownloadHeaders": {"Authorization": "<BEARER TOKEN TOP SECRET>"},
                "modelAuth": {
                    "s3": {
                        "aws_access_key_id": "<SOME ACCESS KEY ID>",
                        "aws_secret_access_key": "<SOME SECRET ACCESS KEY>"
                    }
                }
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": "do not put secrets",
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_skip_image_download_headers(self, mock_time):
        """Test that vectors (custom vector and context) in the query are not logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]

            search_query = {
                "q": "do not put secrets",
                "searchMethod": "TENSOR",
                "limit": 10,
                "image_download_headers": {"Authorization": "<BEARER TOKEN TOP SECRET>"},
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            expected_query = {
                "q": "do not put secrets",
                "searchMethod": "TENSOR",
                "limit": 10,
            }

            self.assertIn(f"{expected_query}", warning_call)

    @patch.dict(os.environ, {
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    @patch('marqo.tensor_search.telemetry.time')
    def test_combination_of_all_fields_hybrid(self, mock_time):
        """Test that all non-secret fields are sanitised and logged"""
        # Reload the module to apply the env vars
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        with patch('marqo.core.search.query_logger.marqo_query_logger') as mock_marqo_query_logger:
            mock_time.perf_counter.side_effect = [0.0, 1.0]
            search_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "offset": 20,
                "rerankDepth": 100,
                "efSearch": 5000,
                "approximate": True,
                "approximateThreshold": 0.5,
                "showHighlights": False,
                "reRanker": "owl/ViT-B/32",
                "filter": "color:red AND brand:Marqo",
                "mediaDownloadHeaders": {"Authorization": "<BEARER TOKEN TOP SECRET>"},
                "modelAuth": {
                    "s3": {
                        "aws_access_key_id": "<SOME ACCESS KEY ID>",
                        "aws_secret_access_key": "<SOME SECRET ACCESS KEY>"
                    }
                },
                "context": {
                    "tensor": [
                        {"vector": [0.2] * 384, "weight": 0.2},
                        {"vector": [0.3] * 384, "weight": 0.8},
                    ],
                    "documents": {
                        "ids": {
                            "doc1": -1.5,
                            "doc2": 0.5
                        },
                        "parameters": {
                            "tensorFields": ["description"],
                            "excludeInputDocuments": False
                        }
                    }
                },
                "textQueryPrefix": "prefix",
                "hybridParameters": {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.3,
                    "rrfK": 60,
                    "searchableAttributesLexical": ["description"],
                    "searchableAttributesTensor": ["description"],
                    "scoreModifiersLexical": {"add_to_score": [{"field_name": "epoch_timestamp", "weight": 0.01}]},
                    "scoreModifiersTensor": {"add_to_score": [{"field_name": "epoch_timestamp", "weight": 0.01}]},
                    "queryLexical": "short lexical",
                    "queryTensor": {
                        "this is a long query with more than 20 characters": 0.3,
                        "this is a short one": 0.2,
                        "and this is another long one": 0.5
                    },
                },
                "facets": {
                    "fields": {
                        "color": {"type": "string", "excludeTerms": ["color:red"]},
                        "brand": {"type": "string", "maxResults": 10, "excludeTerms": ["brand:Marqo"]},
                        "category": {"type": "string", "order": "asc", "maxResults": 5}
                    },
                    "maxDepth": 1000,
                    "maxResults": 3,
                    "order": "desc"
                },
                "trackTotalHits": True,
                "language": "pt",
                "sortBy": {
                    "fields": [
                        {"fieldName": "price", "order": "desc", "missing": "last"}
                    ],
                    "sort_depth": 200,
                    "min_sort_candidates": 500
                },
                "relevanceCutoff": {
                    "method": "mean_std_dev",
                    "probe_depth": 500,
                    "parameters": {"std_dev_factor": 0.5},
                },
                "interpolationMethod": InterpolationMethod.NLERP
            }

            # Execute
            response = self.client.post(f"/indexes/{self.index_name}/search", json=search_query)

            # Verify
            self.assertEqual(response.status_code, 200)
            mock_marqo_query_logger.warning.assert_called_once()
            warning_call = mock_marqo_query_logger.warning.call_args[0][0]
            self.assertIn("Slow search query detected: 1000.0ms", warning_call)
            self.assertIn("Query:", warning_call)

            # Please note that the order of the field must be the same as defined in the pydantic model
            # so we can compare the generated string
            expected_query = {
                "searchMethod": "HYBRID",
                "limit": 10,
                "offset": 20,
                "rerankDepth": 100,
                "efSearch": 5000,
                "approximate": True,
                "approximateThreshold": 0.5,
                "showHighlights": False,
                "reRanker": "owl/ViT-B/32",
                "filter": "color:red AND brand:Marqo",
                "context": {
                    "tensor": [
                        {"vector": [], "weight": 0.2},
                        {"vector": [], "weight": 0.8},
                    ],
                    "documents": {
                        "ids": {
                            "doc1": -1.5,
                            "doc2": 0.5
                        },
                        "parameters": {
                            "tensorFields": ["description"],
                            "excludeInputDocuments": False
                        }
                    }
                },
                "textQueryPrefix": "prefix",
                "hybridParameters": {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.3,
                    "rrfK": 60,
                    "searchableAttributesLexical": ["description"],
                    "searchableAttributesTensor": ["description"],
                    "scoreModifiersLexical": {"add_to_score": [{"field_name": "epoch_timestamp", "weight": 0.01}]},
                    "scoreModifiersTensor": {"add_to_score": [{"field_name": "epoch_timestamp", "weight": 0.01}]},
                    "queryLexical": "short lexical",
                    "queryTensor": {
                        "this is a long query...[truncated:20/49]": 0.3,
                        "this is a short one": 0.2,
                        "and this is another ...[truncated:20/28]": 0.5
                    },
                },
                "facets": {
                    "fields": {
                        "color": {"type": "string", "excludeTerms": ["color:red"]},
                        "brand": {"type": "string", "maxResults": 10, "excludeTerms": ["brand:Marqo"]},
                        "category": {"type": "string", "order": "asc", "maxResults": 5}
                    },
                    "maxDepth": 1000,
                    "maxResults": 3,
                    "order": "desc"
                },
                "trackTotalHits": True,
                "language": "pt",
                "sortBy": {
                    "fields": [
                        {"fieldName": "price", "order": "desc", "missing": "last"}
                    ],
                    "sortDepth": 200,
                    "minSortCandidates": 500
                },
                "relevanceCutoff": {
                    "method": "mean_std_dev",
                    "probeDepth": 500,
                    "parameters": {"stdDevFactor": 0.5},
                    "applyInRetrieval": "both"
                },
                "interpolationMethod": "nlerp"
            }

            query_index = warning_call.find('Query: ')
            self.assertNotEquals(-1, query_index)
            self.assertEqual(f"Query: {expected_query}", warning_call[query_index:])

if __name__ == '__main__':
    unittest.main()