import unittest
from unittest.mock import patch, Mock

from fastapi.responses import ORJSONResponse
from starlette.testclient import TestClient

from marqo.tensor_search import api
from marqo.tensor_search.api import get_documents_by_ids_via_get


class TestApiInitialisation(unittest.TestCase):
    @patch("marqo.tensor_search.api.bootstrap_otel")
    def test_lifespan_integration_bootstrap_and_shutdown_otel(self, mock_bootstrap_otel):
        mock_otel_shutdown_hook = Mock()
        mock_bootstrap_otel.return_value = mock_otel_shutdown_hook

        # Use FastAPI TestClient to simulate making a request to the app
        with TestClient(api.app) as _:
            # Ensure the shutdown hook was called and Zookeeper stop method was triggered
            mock_bootstrap_otel.assert_called_once_with(api.app, service_name='marqo-api')

        mock_otel_shutdown_hook.assert_called_once()


class TestApiGetDocumentEndpoints(unittest.TestCase):
    """Test that get document endpoints return ORJSONResponse"""

    def setUp(self):
        mock_config = Mock()
        self.config_patcher = patch("marqo.tensor_search.api.get_config", return_value=mock_config)
        self.config_patcher.start()
        # Override the FastAPI dependency
        api.app.dependency_overrides[api.get_config] = lambda: mock_config
        self.client = TestClient(api.app, raise_server_exceptions=False)
        self.mock_config = mock_config

    def tearDown(self):
        self.config_patcher.stop()
        api.app.dependency_overrides.clear()

    @patch("marqo.tensor_search.api.tensor_search.get_document_by_id")
    def test_get_document_by_id_returns_orjson_response(self, mock_get_doc):
        """Test that GET /indexes/{index}/documents/{id} returns ORJSONResponse"""
        mock_get_doc.return_value = {"_id": "doc1", "title": "test"}

        resp = self.client.get("/indexes/test_index/documents/doc1")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"_id": "doc1", "title": "test"})
        self.assertIn("application/json", resp.headers["content-type"])

    @patch("marqo.tensor_search.api.tensor_search.get_documents_by_ids")
    def test_get_documents_by_ids_via_get_returns_orjson_response(self, mock_get_docs):
        """Test that GET /indexes/{index}/documents returns ORJSONResponse"""
        mock_result = Mock()
        mock_result.dict.return_value = {"results": [{"_id": "doc1", "_found": True}], "errors": False}
        mock_result.get_header_dict.return_value = {}
        mock_get_docs.return_value = mock_result

        response = get_documents_by_ids_via_get(
            index_name="test_index",
            document_ids=["doc1"],
            marqo_config=self.mock_config,
            expose_facets=False
        )

        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 200)

    @patch("marqo.tensor_search.api.tensor_search.get_documents_by_ids")
    def test_get_documents_by_ids_via_post_returns_orjson_response(self, mock_get_docs):
        """Test that POST /indexes/{index}/documents/get-batch returns ORJSONResponse"""
        mock_result = Mock()
        mock_result.dict.return_value = {"results": [{"_id": "doc1", "_found": True}], "errors": False}
        mock_result.get_header_dict.return_value = {}
        mock_get_docs.return_value = mock_result

        resp = self.client.post(
            "/indexes/test_index/documents/get-batch",
            json={"documentIds": ["doc1"]}
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"][0]["_id"], "doc1")
