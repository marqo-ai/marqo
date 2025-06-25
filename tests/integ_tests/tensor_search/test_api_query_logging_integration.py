import logging
import os
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

    default_env_vars = {
        EnvVars.MARQO_ENABLE_THROTTLING: 'FALSE'  # disable throttling
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        legacy_unstructured_index_v212_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            marqo_version='2.12.0'
        )

        structured_index_request = cls.structured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
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
                docs=[{'_id': '1', 'text_field_1': 'hello'}],
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
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1",  # Very low threshold for testing
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    } | default_env_vars)
    def test_slow_query_logging(self):
        """Integration test for slow query logging with details enabled"""
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
                    self.assertTrue(len(warning_logs) > 0, f"Expected slow query log, but got logs: {self.log_messages}")

                    warning_log = warning_logs[0]
                    self.assertIn(f"Query: {search_query}", warning_log)

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1000",  # High threshold
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    } | default_env_vars)
    def test_fast_query_no_logging(self):
        """Integration test to verify fast queries are not logged"""
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
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1",  # Low threshold
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
    } | default_env_vars)
    def test_slow_query_no_logging_when_disabled(self):
        """Integration test to verify fast queries are not logged"""
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
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    } | default_env_vars)
    def test_search_error_logging(self):
        """Integration test for search error logging"""
        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "filter": "error_filter",
                        "limit": 1,
                        "searchMethod": search_method.value
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
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "500",
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "FALSE"
    } | default_env_vars)
    def test_search_error_no_logging_when_disabled(self):
        """Integration test for search error logging"""
        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "filter": "error_filter",
                        "limit": 1,
                        "searchMethod": search_method.value
                    }

                    # Execute
                    response = self.client.post(f"/indexes/{index.name}/search", json=search_query)

                    self.assertEqual(response.status_code, 400)

                    self.assertTrue(len(self.log_messages) == 0,
                                    f"Expected no query log, but got logs: {self.log_messages}")

    @patch.dict(os.environ, {
        EnvVars.MARQO_VESPA_SLOW_QUERY_THRESHOLD_MS: "1",
        EnvVars.MARQO_VESPA_LOG_QUERY_DETAILS: "TRUE"
    } | default_env_vars)
    def test_search_error_logging_overrides_slow_query_logging(self):
        """Integration test for search error logging"""
        for search_method in SearchMethod:
            for index in self.indexes:
                with self.subTest(search_method=search_method, index=index.type):
                    self._setup_log_capture()

                    search_query = {
                        "q": "hello",
                        "filter": "error_filter",
                        "limit": 1,
                        "searchMethod": search_method.value
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