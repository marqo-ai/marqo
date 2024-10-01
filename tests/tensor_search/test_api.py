import uuid
from unittest import mock
from unittest.mock import patch

from fastapi.testclient import TestClient

import marqo.tensor_search.api as api
from marqo import exceptions as base_exceptions
from marqo.api.exceptions import BadRequestError
from marqo.core import exceptions as core_exceptions
from marqo.core.exceptions import ZeroMagnitudeVectorError
from marqo.core.models.marqo_index import FieldType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search.enums import EnvVars
from marqo.vespa import exceptions as vespa_exceptions
from tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
import importlib
import sys
import os


class ApiTests(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_add_or_replace_documents_tensor_fields(self):
        with mock.patch('marqo.tensor_search.tensor_search.add_documents') as mock_add_documents:
            mock_add_documents.return_value = MarqoAddDocumentsResponse(
                errors=False,
                processingTimeMs=0.0,
                index_name="index1",
                items=[
                    MarqoAddDocumentsItem(
                        status=200,
                        id="1",
                    )
                ],
            )
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

    def test_memory(self):
        """
        Test that the memory endpoint returns the expected keys when debug API is enabled.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_DEBUG_API: 'TRUE'}):
            response = self.client.get("/memory")
            data = response.json()
            assert set(data.keys()) == {"memory_used", "stats"}

    def test_memory_defaultDisabled(self):
        """
        Test that the memory endpoint returns 403 by default.
        """
        response = self.client.get("/memory")
        self.assertEqual(response.status_code, 403)

    def test_memory_disabled_403(self):
        """
        Test that the memory endpoint returns 403 when debug API is disabled explicitly.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_DEBUG_API: 'FALSE'}):
            response = self.client.get("/memory")
            self.assertEqual(response.status_code, 403)

    def test_custom_search_limit(self):
        """
        Test that the search endpoint returns the expected search limit when MARQO_MAX_SEARCH_LIMIT is set.
        """
        custom_limits = [2000, 1000000]
        for custom_limit in custom_limits:
            with patch.dict('os.environ', {
                EnvVars.MARQO_MAX_SEARCH_LIMIT: str(custom_limit),
                EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(custom_limit + 1000000)
            }):
                response = self.client.post(
                    "/indexes/index1/search?device=cpu",
                    json={
                        "q": "test",
                        "searchMethod": "TENSOR",
                        "limit": custom_limit + 1,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn(f"result limit must be less than or equal to the "
                                f"MARQO_MAX_SEARCH_LIMIT limit of [{custom_limit}]",
                                response.json()["message"])

    def test_custom_search_offset(self):
        """
        Test that the search endpoint returns the expected search limit when MARQO_MAX_SEARCH_OFFSET is set.
        """
        custom_offsets = [2000, 1000000]
        for custom_offset in custom_offsets:
            with patch.dict('os.environ', {
                EnvVars.MARQO_MAX_SEARCH_OFFSET: str(custom_offset),
                EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(custom_offset + 1000000)
            }):
                response = self.client.post(
                    "/indexes/index1/search?device=cpu",
                    json={
                        "q": "test",
                        "searchMethod": "TENSOR",
                        "offset": custom_offset + 1,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn(f"The search result offset must be less than or equal "
                                f"to the MARQO_MAX_SEARCH_OFFSET limit of [{custom_offset}]",
                                response.json()["message"])

class ValidationApiTests(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_schema_validation_defaultDisabled(self):
        """
        Test that the schema_validation endpoint returns 403 by default.
        """
        data = {
            "type": "structured",
            "allFields": [],
            "tensorFields": []
        }
        index_name = "test-index"
        response = self.client.post(f"/validate/index/{index_name}", json=data)
        self.assertEqual(response.status_code, 403)

    def test_ops_api_disabled_403(self):
        """
        Test that the ops-api endpoint returns 403 when debug API is disabled explicitly.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'FALSE'}):
            data = {
                "type": "structured",
                "allFields": [],
                "tensorFields": [],
                "settings_object": {}
            }
            index_name = "test-index"
            response = self.client.post(f"/validate/index/{index_name}", json=data)
            self.assertEqual(response.status_code, 403)

    def test_ops_api_200(self):
        """
        Test that the ops-api endpoint returns 200 when debug API is enabled.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'TRUE'}):
            data = {
                "treatUrlsAndPointersAsImages": False,
                "model": "hf/e5-large",
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence",
                },
                "imagePreprocessing": {"patchMethod": None},
                "annParameters": {
                    "spaceType": "euclidean",
                    "parameters": {"efConstruction": 128, "m": 16},
                },
                "type": "unstructured",
            }
            index_name = "test-index"
            response = self.client.post(f"/validate/index/{index_name}", json=data)
            self.assertEqual(response.json(), {'validated': True, 'index': 'test-index'})

    def test_ops_api_400(self):
        """
        Test that the ops-api endpoint returns 400 when debug API is enabled and the input is invalid.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'TRUE'}):
            data = {
                "treatUrlsAndPointersAsImages": False,
                "model": "hf/e5-large",
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence",
                },
                "imagePreprocessing": {"patchMethod": None},
                "annParameters": {
                    "spaceType": "euclidean",
                    "parameters": {"efConstruction": 128, "m": 16},
                },
                "type": "unknown"  # invalid type
            }
            index_name = "test-index"
            response = self.client.post(f"/validate/index/{index_name}", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn("message", response.json())
            self.assertEqual(response.json()["code"], "invalid_argument")
            self.assertEqual(response.json()["type"], "invalid_request")


class TestApiCustomEnvVars(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request()
        structured_index_request = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='field1', type=FieldType.Text, features=['lexical_search']),
                FieldRequest(name='field2', type=FieldType.Text)
            ],
            tensor_fields=['field1']
        )

        cls.indexes = cls.create_indexes([unstructured_index_request, structured_index_request])

        cls.unstructured_index = cls.indexes[0]
        cls.structured_index = cls.indexes[1]

    def test_search_timeout_short_timer_fails(self):
        # Set up the test API client with the correct env vars set
        with mock.patch.dict(os.environ, {"VESPA_SEARCH_TIMEOUT_MS": "1"}):
            importlib.reload(sys.modules['marqo.tensor_search.api'])
            # VespaClient will be created with default timeout of 1ms
            self.client = TestClient(api.app)

            with self.subTest(search_method="TENSOR"):
                for index in [self.unstructured_index, self.structured_index]:
                    with self.subTest(index=index.name):
                        res = self.client.post("/indexes/" + index.name + "/search?device=cpu", json={
                            "q": "irrelevant",
                            "searchMethod": "TENSOR"
                        })
                        # The search request must timeout, since the timeout is set to 1ms
                        self.assertEqual(res.status_code, 504)
                        self.assertEqual(res.json()["code"], "vector_store_timeout")
                        self.assertEqual(res.json()["type"], "invalid_request")

            with self.subTest(search_method="HYBRID"):
                for index in [self.structured_index]:   # TODO: add unstructured when supported
                    with self.subTest(index=index.name):
                        res = self.client.post("/indexes/" + index.name + "/search?device=cpu", json={
                            "q": "irrelevant",
                            "searchMethod": "HYBRID"
                        })
                        # The search request must timeout, since the timeout is set to 1ms
                        self.assertEqual(res.status_code, 504)
                        self.assertEqual(res.json()["code"], "vector_store_timeout")
                        self.assertEqual(res.json()["type"], "invalid_request")


class TestApiErrors(MarqoTestCase):
    """
    Execute requests that trigger core/base errors.
    Handler should return the correct API error, even if the internal function raises a base error.

    Testing on errors that should be 4xxs.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request()
        structured_index_request = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='field1', type=FieldType.Text),
                FieldRequest(name='field2', type=FieldType.Text)
            ],
            tensor_fields=['field1']
        )

        cls.indexes = cls.create_indexes([unstructured_index_request, structured_index_request])

        cls.unstructured_index = cls.indexes[0]
        cls.structured_index = cls.indexes[1]

    def setUp(self):
        self.client = TestClient(api.app)

    def test_index_not_found_error(self):
        index_name = self.random_index_name()

        response = self.client.delete("/indexes/" + index_name)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "index_not_found")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "not found" in response.json()["message"] and index_name in response.json()["message"]

    def test_index_already_exists(self):
        response = self.client.post("/indexes/" + self.structured_index.name, json={
            "type": "structured",
            "allFields": [],
            "tensorFields": []
        })

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "index_already_exists")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "already exists" in response.json()["message"] and self.structured_index.name in response.json()[
            "message"]

    def test_invalid_field_name(self):
        # use attributesToRetrieve on a non-existent field
        response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
            "q": "test",
            "attributesToRetrieve": ["non_existent_field"]
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_field_name")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "has no field non_existent_field" in response.json()["message"]

    def test_invalid_data_type(self):
        """Test that invalid data types only reject the document with the invalid data type, not the whole request"""
        # Add a document to field1 of the wrong type
        response = self.client.post("/indexes/" + self.structured_index.name + "/documents?device=cpu", json={
            "documents": [
                {
                    "field2": 123
                }
            ]
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["errors"], True)
        self.assertIn("Expected a value of type", response.json()["items"][0]["error"])

    def test_filter_string_parsing_error(self):
        response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
            "q": "test",
            "filter": ""
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_argument")
        self.assertEqual(response.json()["type"], "invalid_request")
        assert "Cannot parse empty filter string" in response.json()["message"]

    def test_vespa_timeout_error(self):
        error = vespa_exceptions.VespaTimeoutError('timeout_msg')
        with patch("marqo.tensor_search.tensor_search.search", side_effect=error):
            response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
                "q": "test",
                "filter": ""
            })

            self.assertEqual(response.status_code, 504)
            self.assertEqual(response.json()["code"], "vector_store_timeout")
            self.assertEqual(response.json()["type"], "invalid_request")
            assert "Vector store request timed out" in response.json()["message"]

    def test_invalid_argument_error(self):
        # Try to create index with invalid model (should raise 400)
        response = self.client.post("/indexes/" + self.random_index_name(), json={
            "type": "structured",
            "allFields": [{"name": "field1", "type": "text"}],
            "tensorFields": [],
            "model": "random_nonexistent_model"
        })

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
                response = self.client.post(
                    "/indexes/my_index",
                    json=test_case
                )

                self.assertEqual(response.status_code, 422)
                self.assertTrue(f"Invalid field name '{field}'" in response.text)

        for test_case, test_name in test_cases_pass:
            with self.subTest(test_name):
                index_name = 'a' + str(uuid.uuid4()).replace('-', '')
                response = self.client.post(
                    f"/indexes/{index_name}",
                    json=test_case
                )

                self.assertEqual(response.status_code, 200)

    def test_invalid_structured_index_field_type(self):
        """Verify invalid field types are rejected with proper error"""

        base_index_settings = {
            "type": "structured",
            "allFields": [{"name": "field1", "type": None}],
            "tensorFields": []
        }

        test_cases = [
            ("bulabua", "Invalid field type 'bulabua'"),
            ([], "Invalid field type '[]'"),
            (None, "Invalid field type 'NoneType'"),
            ("", "Invalid field type ''"),
        ]

        for test_case, test_name in test_cases:
            test_settings = base_index_settings.copy()
            test_settings["allFields"][0]["type"] = test_case
            with self.subTest(test_name):
                index_name = 'a' + str(uuid.uuid4()).replace('-', '')
                response = self.client.post(
                    f"/indexes/{index_name}",
                    json=test_settings
                )
                self.assertEqual(response.status_code, 422)
                self.assertIn("allFields", response.text)
                self.assertIn("type", response.text)

    def test_invalid_structured_index_field_features(self):
        """Verify invalid field features are rejected with proper error"""

        base_index_settings = {
            "type": "structured",
            "allFields": [{"name": "field1", "type": "text", "features": None}],
            "tensorFields": []
        }

        test_cases = [
            ("bulabua", "Invalid field feature 'bulabua'"),
            (None, "Invalid field feature 'NoneType'"),
            ("", "Invalid field feature ''"),
        ]

        for test_case, test_name in test_cases:
            test_settings = base_index_settings.copy()
            test_settings["allFields"][0]["features"] = test_case
            with self.subTest(test_name):
                index_name = 'a' + str(uuid.uuid4()).replace('-', '')
                response = self.client.post(
                    f"/indexes/{index_name}",
                    json=test_settings
                )
                self.assertEqual(response.status_code, 422)
                self.assertIn("allFields", response.text)
                self.assertIn("features", response.text)

    def test_log_stack_trace_for_core_exceptions(self):
        """Ensure stack trace is logged for core exceptions, e.g.,IndexExistsError"""
        raised_error = core_exceptions.IndexExistsError("index1")
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with patch("marqo.core.index_management.index_management.IndexManagement.create_index",
                       side_effect=raised_error):
                response = self.client.post("/indexes/" + self.structured_index.name, json={
                    "type": "structured",
                    "allFields": [{"name": "field1", "type": "text"}],
                    "tensorFields": [],
                })
            mock_logger_error.assert_called_once()
            self.assertIn("index1", str(mock_logger_error.call_args))

    def test_log_stack_trace_for_base_exceptions_invalid_arg(self):
        """Ensure stack trace is logged for base exceptions, e.g.,InvalidArg"""
        raised_error = base_exceptions.InvalidArgumentError("invalid_arg_msg")
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with patch("marqo.tensor_search.tensor_search.search", side_effect=raised_error):
                response = self.client.post(f"/indexes/test_index/search", json={
                    "q": "test"
                })
            mock_logger_error.assert_called_once()
            self.assertIn("invalid_arg_msg", str(mock_logger_error.call_args))

    def test_log_stack_trace_for_base_exceptions_internal(self):
        """Ensure stack trace is logged for base exceptions, e.g.,InternalError"""
        raised_error = base_exceptions.InternalError("internal_error_msg")
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with patch("marqo.tensor_search.tensor_search.get_document_by_id", side_effect=raised_error):
                response = self.client.get(f"/indexes/test_index/documents/1")
            mock_logger_error.assert_called_once()
            self.assertIn("internal_error_msg", str(mock_logger_error.call_args))

    # TODO: Test how marqo handles generic exceptions, including Exception, RunTimeError, ValueError, etc.
