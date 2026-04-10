import importlib
import logging
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

import marqo.tensor_search.api as api
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model, FieldType, FieldFeature, IndexType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search.enums import EnvVars, SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestAPIQueryLoggingIntegration(MarqoTestCase):
    """Integration tests for the query logging feature in the API"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        legacy_unstructured_index_v212_request = cls.unstructured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
            marqo_version='2.12.0'
        )

        structured_index_request = cls.structured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])],
            tensor_fields=["text_field_1"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_index_request,
            legacy_unstructured_index_v212_request,
            structured_index_request,
        ])

        for index in cls.indexes:
            cls.add_documents(cls.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=[
                    {'_id': 'doc1', 'text_field_1': 'hello'},
                    {'_id': 'doc2', 'text_field_1': 'world'},
                    {'_id': 'doc3', 'text_field_1': 'hello world'},
                ],
                tensor_fields=None if index.type == IndexType.Structured else ["text_field_1"]
            ))

    def setUp(self):
        self.client = TestClient(api.app)

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

        logger = logging.getLogger('marqo_query')

        # clean up logs from the last run
        if self.log_handler:
            logger.removeHandler(self.log_handler)
            self.log_messages = []

        self.log_handler = LogCapture(self.log_messages)
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1",  # Very low threshold for testing
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    def test_slow_query_logging(self):
        """Integration test for slow query logging with details enabled"""

        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "searchMethod": search_method.value,
                        "limit": 1,
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    # Verify response is successful
                    self.assertEqual(response.status_code, 200)

                    # Verify slow query was logged with details
                    warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
                    self.assertTrue(len(warning_logs) > 0,
                                    f"Expected slow query log, but got logs: {self.log_messages}")

                    warning_log = warning_logs[0]
                    self.assertIn(f"Query: {search_query}", warning_log)

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1",  # Very low threshold for testing
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    def test_slow_query_logging_all_fields_sanitised_excluding_secret_fields(self):
        """Integration test for slow query logging with all fields for a hybrid query on semistructured index"""

        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])
        search_method = SearchMethod.HYBRID
        index = self.indexes[0]

        self._setup_log_capture()

        search_query = {
            "searchMethod": search_method.value,
            "limit": 1,
            "offset": 20,
            "rerankDepth": 100,
            "efSearch": 5000,
            "approximate": True,
            "approximateThreshold": 0.5,
            "showHighlights": False,
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
                        "tensorFields": ["text_field_1"],
                        "excludeInputDocuments": False
                    }
                }
            },
            "hybridParameters": {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.3,
                "rrfK": 60,
                "searchableAttributesLexical": ["text_field_1"],
                "searchableAttributesTensor": ["text_field_1"],
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
                "sortDepth": 200,
                "minSortCandidates": 500
            },
            "relevanceCutoff": {
                "method": "mean_std_dev",
                "probeDepth": 500,
                "parameters": {"stdDevFactor": 0.5}
            },
            "interpolationMethod": "nlerp"
        }

        # Execute
        response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

        # Verify response is successful
        self.assertEqual(response.status_code, 200)

        # Verify slow query was logged with details
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertTrue(len(warning_logs) > 0,
                        f"Expected slow query log, but got logs: {self.log_messages}")

        expected_query = {
            "searchMethod": "HYBRID",
            "limit": 1,
            "offset": 20,
            "rerankDepth": 100,
            "efSearch": 5000,
            "approximate": True,
            "approximateThreshold": 0.5,
            "showHighlights": False,
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
                        "tensorFields": ["text_field_1"],
                        "excludeInputDocuments": False
                    }
                }
            },
            "hybridParameters": {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.3,
                "rrfK": 60,
                "searchableAttributesLexical": ["text_field_1"],
                "searchableAttributesTensor": ["text_field_1"],
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
        query_index = warning_logs[0].find('Query: ')
        self.assertNotEquals(-1, query_index)
        self.assertEqual(f"Query: {expected_query}", warning_logs[0][query_index:])

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1",  # Very low threshold for testing
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE",
        EnvVars.MARQO_LOG_QUERY_MAX_LENGTH: "20"
    })
    def test_slow_query_logging_sanitised_custom_vector_fields(self):
        """Integration test for slow query logging with all fields for a hybrid query on semistructured index"""

        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])
        search_method = SearchMethod.HYBRID
        index = self.indexes[0]

        self._setup_log_capture()

        search_query = {
            "q": {
                "customVector": {
                    "content": "this is a long query with more than 20 characters",
                    "vector": [0.1] * 384
                }
            },
            "searchMethod": search_method.value,
            "limit": 1,
        }

        # Execute
        response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

        # Verify response is successful
        self.assertEqual(response.status_code, 200)

        # Verify slow query was logged with details
        warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
        self.assertTrue(len(warning_logs) > 0,
                        f"Expected slow query log, but got logs: {self.log_messages}")

        expected_query = {
            "q": {
                "customVector": {
                    "content": "this is a long query...[truncated:20/49]",
                    "vector": []
                }
            },
            "searchMethod": "HYBRID",
            "limit": 1,
        }
        self.assertIn(f"Query: {expected_query}", warning_logs[0])

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1000",  # High threshold
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    def test_fast_query_no_logging(self):
        """Integration test to verify fast queries are not logged"""
        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "limit": 1,
                        "searchMethod": search_method.value
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    # Verify response is successful
                    self.assertEqual(response.status_code, 200)

                    # Verify slow query was logged with details
                    warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
                    self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs, but got: {warning_logs}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1",  # Low threshold
        EnvVars.MARQO_LOG_QUERY_DETAILS: "FALSE"
    })
    def test_slow_query_no_logging_when_disabled(self):
        """Integration test to verify fast queries are not logged"""
        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "limit": 1,
                        "searchMethod": search_method.value
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    # Verify response is successful
                    self.assertEqual(response.status_code, 200)

                    # Verify slow query was nog logged
                    warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
                    self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs, but got: {warning_logs}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "500",
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    def test_search_error_logging(self):
        """Integration test for search error logging"""
        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "searchMethod": search_method.value,
                        "limit": 1,
                        "filter": "error_filter",
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    self.assertEqual(response.status_code, 400)

                    # Verify slow query was logged with details
                    error_logs = [msg for msg in self.log_messages if "Failed search query" in msg]
                    self.assertTrue(len(error_logs) > 0,
                                    f"Expected failed query log, but got logs: {self.log_messages}")

                    self.assertIn(f"Query: {search_query}", error_logs[0])

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "500",
        EnvVars.MARQO_LOG_QUERY_DETAILS: "FALSE"
    })
    def test_search_error_no_logging_when_disabled(self):
        """Integration test for search error logging"""
        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "searchMethod": search_method.value,
                        "limit": 1,
                        "filter": "error_filter",
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    self.assertEqual(response.status_code, 400)

                    self.assertTrue(len(self.log_messages) == 0,
                                    f"Expected no query log, but got logs: {self.log_messages}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS: "1",
        EnvVars.MARQO_LOG_QUERY_DETAILS: "TRUE"
    })
    def test_search_error_logging_overrides_slow_query_logging(self):
        """Integration test for search error logging"""
        # reload the module to apply the env var change
        importlib.reload(sys.modules['marqo.core.search.query_logger'])

        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "searchMethod": search_method.value,
                        "limit": 1,
                        "filter": "error_filter",
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    self.assertEqual(response.status_code, 400)

                    # Verify slow query was logged with details
                    error_logs = [msg for msg in self.log_messages if "Failed search query" in msg]
                    self.assertTrue(len(error_logs) > 0,
                                    f"Expected failed query log, but got logs: {self.log_messages}")

                    self.assertIn(f"Query: {search_query}", error_logs[0])

                    warning_logs = [msg for msg in self.log_messages if "Slow search query detected" in msg]
                    self.assertEqual(len(warning_logs), 0, f"Expected no slow query logs, but got: {warning_logs}")


if __name__ == '__main__':
    import unittest

    unittest.main()
