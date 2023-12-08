from unittest import mock

from fastapi.testclient import TestClient

import marqo.tensor_search.api as api
from tests.marqo_test import MarqoTestCase


class ApiTests(MarqoTestCase):
    def setUp(self):
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


class TestApiErrors(MarqoTestCase):
    """
    Execute requests that trigger core/base errors.
    Handler should return the correct API error, even if the internal function raises a base error.

    Testing on errors that should be 4xxs.
    """

    index_name_1 = "index1"
    def setUp(self):
        self.client = TestClient(api.app)
        self.index_name_1 = "index1"

    def tearDown(self) -> None:
        # Make sure no indexes are left over from tests
        self.client.delete("/indexes/" + self.index_name_1)

    def test_index_not_found_error(self):
        # delete index if it exists
        self.client.delete("/indexes/" + self.index_name_1)

        response = self.client.delete("/indexes/" + self.index_name_1)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "index_not_found")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "does not exist" in response.json()["message"] and "index1" in response.json()["message"]

    def test_index_already_exists(self):
        # create index if it does not already exist
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [],
            "tensor_fields": []
        })
        response = self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [],
            "tensor_fields": []
        })

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "index_already_exists")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "already exists" in response.json()["message"] and "index1" in response.json()["message"]

    def test_invalid_field_name(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [],
            "tensor_fields": []
        })

        # use attributes_to_retrieve on a non-existent field
        response = self.client.post("/indexes/" + self.index_name_1 + "/search?device=cpu", json={
            "q": "test",
            "attributesToRetrieve": ["non_existent_field"]
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_field_name")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "has no field non_existent_field" in response.json()["message"]

    def test_invalid_data_type(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [{"name": "field1", "type": "text"}],
            "tensor_fields": []
        })

        # Add a document to field1 of the wrong type
        response = self.client.post("/indexes/" + self.index_name_1 + "/documents?device=cpu", json={
            "documents": [
                {
                    "field1": 123
                }
            ]
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Expected a value of" in response.json()["message"] and "but found" in response.json()["message"]

    def test_filter_string_parsing_error(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [{"name": "field1", "type": "text"}],
            "tensor_fields": []
        })

        response = self.client.post("/indexes/" + self.index_name_1 + "/search?device=cpu", json={
            "q": "test",
            "filter": ""
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Cannot parse empty filter string" in response.json()["message"]

    def test_invalid_argument_error(self):
        # Try to create index with invalid model (should raise 400)
        response = self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "all_fields": [{"name": "field1", "type": "text"}],
            "tensor_fields": [],
            "model": "random_nonexistent_model"
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Could not find model properties for" in response.json()["message"]
