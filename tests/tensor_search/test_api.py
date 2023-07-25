from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import marqo.tensor_search.api as api
from tests.marqo_test import MarqoTestCase


class ApiTests(MarqoTestCase):
    def setUp(self):
        api.OPENSEARCH_URL = 'http://localhost:0000'
        self.client = TestClient(api.app)

    def test_add_or_replace_documents_tensor_fields(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu",
                json={
                    "documents": [
                        {
                            "id": "1",
                            "text": "This is a test document",
                        }
                    ],
                    "tensorFields": ['text']
                },
            )
            self.assertEqual(response.status_code, 200)
            mock_add_documents.assert_called_once()

    def test_add_or_replace_documents_non_tensor_fields(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu",
                json={
                    "documents": [
                        {
                            "id": "1",
                            "text": "This is a test document",
                        }
                    ],
                    "nonTensorFields": ['text']
                },
            )
            self.assertEqual(response.status_code, 200)
            mock_add_documents.assert_called_once()

    def test_add_or_replace_documents_non_tensor_fields_query_param(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu&non_tensor_fields=['text']",
                json=[
                    {
                        "id": "1",
                        "text": "This is a test document",
                    }
                ]
            )
            self.assertEqual(response.status_code, 200)
            mock_add_documents.assert_called_once()

    def test_add_or_replace_documents_tensor_fields_undefined_body(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu",
                json={
                    "documents": [
                        {
                            "id": "1",
                            "text": "This is a test document",
                        }
                    ]
                },
            )
            self.assertEqual(response.status_code, 400)
            mock_add_documents.assert_not_called()

    def test_add_or_replace_documents_fields_undefined_query_param(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu",
                json=
                [
                    {
                        "id": "1",
                        "text": "This is a test document",
                    }
                ]
            )
            self.assertEqual(response.status_code, 400)
            mock_add_documents.assert_not_called()
