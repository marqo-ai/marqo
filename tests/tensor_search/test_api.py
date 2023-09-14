from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import marqo.tensor_search.api as api
from tests.marqo_test import MarqoTestCase


class ApiTestsAddDocs(MarqoTestCase):
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
                "/indexes/index1/documents?device=cpu&non_tensor_fields=text&non_tensor_fields=title",
                json=[
                    {
                        "id": "1",
                        "title": "My doc",
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
    
    def test_add_or_replace_documents_defaults(self):
        """
        Ensures that API calls to add or replace documents call tensor_search.add_documents
        with the correct defaults (eg. auto_refresh)
        """
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
            args, kwargs = mock_add_documents.call_args

            # Assert that add documents is called with the correct default arguments
            assert kwargs["add_docs_params"].auto_refresh == False
            assert kwargs["add_docs_params"].use_existing_tensors == False
    
    def test_add_or_replace_documents_auto_refresh_true(self):
        """
        Ensures that calling add documents with some parameters set to non-default values
        (refresh, use_existing_tensors) works as expected.
        """
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            response = self.client.post(
                "/indexes/index1/documents?device=cpu&refresh=true",
                json={
                    "documents": [
                        {
                            "id": "1",
                            "text": "This is a test document",
                        }
                    ],
                    "useExistingTensors": True,
                    "tensorFields": ['text']
                },
            )

            self.assertEqual(response.status_code, 200)
            mock_add_documents.assert_called_once()
            args, kwargs = mock_add_documents.call_args

            # Assert that add documents is called with the correct new arguments
            assert kwargs["add_docs_params"].auto_refresh == True
            assert kwargs["add_docs_params"].use_existing_tensors == True


class ApiTestsDeleteDocs(MarqoTestCase):
    def setUp(self):
        api.OPENSEARCH_URL = 'http://localhost:0000'
        self.client = TestClient(api.app)

    def test_delete_docs_defaults(self):
        """
        Ensures that API calls to delete documents call tensor_search.delete_documents
        with the correct defaults (eg. auto_refresh)
        """

        with mock.patch('marqo.tensor_search.tensor_search.delete_documents') as mock_delete_documents:
            response = self.client.post(
                "/indexes/index1/documents/delete-batch",
                json=['0', '1', '2']
            )
            """
            TODO: figure out why this format results in an error:
            json={
                "documentIds": ['0', '1', '2']
            }
            """
            
            self.assertEqual(response.status_code, 200)
            mock_delete_documents.assert_called_once()
            args, kwargs = mock_delete_documents.call_args

            # Assert that delete_documents is called with the correct default arguments
            assert kwargs["auto_refresh"] == False
    
    def test_delete_docs_auto_refresh_true(self):
        """
        Ensures that API calls to delete documents with parameters set (auto_refresh=True)
        reflect those in calls to tensor_search.delete_documents
        """

        with mock.patch('marqo.tensor_search.tensor_search.delete_documents') as mock_delete_documents:
            response = self.client.post(
                "/indexes/index1/documents/delete-batch?refresh=true",
                json=['0', '1', '2']
            )
            
            self.assertEqual(response.status_code, 200)
            mock_delete_documents.assert_called_once()
            args, kwargs = mock_delete_documents.call_args

            # Assert that delete_documents is called with the correct new arguments
            assert kwargs["auto_refresh"] == True