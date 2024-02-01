import uuid
from unittest import mock
from unittest.mock import patch

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
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            response = self.client.delete("/indexes/" + self.index_name_1)

            mock_logger_error.assert_called_once()
            self.assertIn("not found", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "index_not_found")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "not found" in response.json()["message"] and "index1" in response.json()["message"]

    def test_index_already_exists(self):
        # create index if it does not already exist
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "allFields": [],
            "tensorFields": []
        })
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            response = self.client.post("/indexes/" + self.index_name_1, json={
                "type": "structured",
                "allFields": [],
                "tensorFields": []
            })
            mock_logger_error.assert_called_once()
            self.assertIn("already exists", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "index_already_exists")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "already exists" in response.json()["message"] and "index1" in response.json()["message"]

    def test_invalid_field_name(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "allFields": [],
            "tensorFields": []
        })
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            # use attributesToRetrieve on a non-existent field
            response = self.client.post("/indexes/" + self.index_name_1 + "/search?device=cpu", json={
                "q": "test",
                "attributesToRetrieve": ["non_existent_field"]
            })
            mock_logger_error.assert_called_once()
            self.assertIn("has no field non_existent_field", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_field_name")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "has no field non_existent_field" in response.json()["message"]

    def test_invalid_data_type(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "allFields": [{"name": "field1", "type": "text"}],
            "tensorFields": []
        })
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            # Add a document to field1 of the wrong type
            response = self.client.post("/indexes/" + self.index_name_1 + "/documents?device=cpu", json={
                "documents": [
                    {
                        "field1": 123
                    }
                ]
            })
            mock_logger_error.assert_called_once()
            self.assertIn("Expected a value of", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Expected a value of" in response.json()["message"] and "but found" in response.json()["message"]

    def test_filter_string_parsing_error(self):
        self.client.post("/indexes/" + self.index_name_1, json={
            "type": "structured",
            "allFields": [{"name": "field1", "type": "text"}],
            "tensorFields": []
        })
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            response = self.client.post("/indexes/" + self.index_name_1 + "/search?device=cpu", json={
                "q": "test",
                "filter": ""
            })
            mock_logger_error.assert_called_once()
            self.assertIn("Cannot parse empty filter string", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Cannot parse empty filter string" in response.json()["message"]

    def test_invalid_argument_error(self):
        # Try to create index with invalid model (should raise 400)
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            response = self.client.post("/indexes/" + self.index_name_1, json={
                "type": "structured",
                "allFields": [{"name": "field1", "type": "text"}],
                "tensorFields": [],
                "model": "random_nonexistent_model"
            })
            mock_logger_error.assert_called_once()
            self.assertIn("Could not find model properties for", str(mock_logger_error.call_args))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Could not find model properties for" in response.json()["message"]

    def test_create_index_snake_case_fails(self):
        """
        Verify snake case rejected for fields that have camel case as alias
        """
        test_cases_fail = [
            ({
                 "type": "structured",
                 "allFields": [
                     {
                         "name": "field1",
                         "type": "text"
                     },
                     {
                         "name": "field2",
                         "type": "text"
                     },
                     {
                         "name": "field3",
                         "type": "multimodal_combination",
                         "dependent_fields": {"field1": 0.5, "field2": 0.5}
                     }
                 ],
                 "tensorFields": [],
             }, 'dependent_fields', 'Snake case within a list'),
            ({
                 "type": "structured",
                 "allFields": [],
                 "tensorFields": [],
                 'annParameters': {
                     'spaceType': 'dotproduct',
                     'parameters': {
                         'ef_construction': 128,
                         'm': 16
                     }
                 }
             }, 'ef_construction', 'Snake case within a dict is invalid'),
            ({
                 "type": "unstructured",
                 'annParameters': {
                     'spaceType': 'dotproduct',
                     'parameters': {
                         'ef_construction': 128,
                         'm': 16
                     }
                 }
             }, 'ef_construction', 'Snake case within a dict is invalid, unstructured index')
        ]
        test_cases_pass = [
            ({
                 "type": "structured",
                 "allFields": [
                     {
                         "name": "field_1",
                         "type": "text"
                     },
                     {
                         "name": "field_2",
                         "type": "text"
                     },
                     {
                         "name": "field_3",
                         "type": "multimodal_combination",
                         "dependentFields": {"field_1": 0.5, "field_2": 0.5}
                     }
                 ],
                 "tensorFields": ['field_3'],
                 "model": "ViT-L/14",
                 "modelProperties": {
                     "name": "ViT-L/14",
                     "dimensions": 768,
                     "url": "https://7b4d1a66-507d-43f1-b99f-7368b655de46.s3.amazonaws.com/e5a7d9c7-0736-4301-a037-b1307f43a314/23fa0cb1-68d5-40f6-8039-e9e1265b6103.pt",
                     "type": "open_clip",
                     "field_1": "sth"
                 }
             }, 'Snake case in field name is valid'),
        ]

        for test_case, field, test_name in test_cases_fail:
            with self.subTest(test_name):
                with patch('marqo.api.route.logger.error') as mock_logger_error:
                    response = self.client.post(
                        "/indexes/my_index",
                        json=test_case
                    )
                    mock_logger_error.assert_called_once()

                self.assertEqual(response.status_code, 422)
                self.assertTrue(f"Invalid field name '{field}'" in response.text)

        for test_case, test_name in test_cases_pass:
            with self.subTest(test_name):
                with patch('marqo.api.route.logger.error') as mock_logger_error:
                    index_name = 'a' + str(uuid.uuid4()).replace('-', '')
                    response = self.client.post(
                        f"/indexes/{index_name}",
                        json=test_case
                    )
                    mock_logger_error.assert_not_called()

                self.assertEqual(response.status_code, 200)
