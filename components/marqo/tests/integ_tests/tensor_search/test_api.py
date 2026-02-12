import importlib
import os
import sys
import uuid
from unittest import mock
from unittest.mock import patch, Mock

import pydantic
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic.v1.error_wrappers import ErrorWrapper
from pydantic_core import InitErrorDetails, PydanticCustomError

from fastapi.responses import ORJSONResponse

import marqo.tensor_search.api as api
from marqo.tensor_search.api import get_documents_by_ids_via_get, get_config
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo import exceptions as base_exceptions
from marqo import version
from marqo.api.exceptions import InvalidArgError
from marqo.core import exceptions as core_exceptions
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_index import FieldType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.inference.inference_cache.caching_inference import CachingInference
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.vespa import exceptions as vespa_exceptions


class ApiTests(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_add_or_replace_documents_tensor_fields(self):
        with mock.patch('marqo.core.document.document.Document.add_documents') as mock_add_documents:
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


class TestApiGetDocumentEndpoints(MarqoTestCase):
    """Integration tests for get document endpoints returning ORJSONResponse"""

    def setUp(self):
        self.client = TestClient(api.app)

    @mock.patch('marqo.tensor_search.tensor_search.get_documents_by_ids')
    def test_get_documents_by_ids_via_get_returns_orjson_response(self, mock_get_docs):
        """Test that GET /indexes/{index}/documents returns ORJSONResponse"""
        mock_result = Mock()
        mock_result.dict.return_value = {"results": [{"_id": "doc1", "_found": True}], "errors": False}
        mock_result.get_header_dict.return_value = {}
        mock_get_docs.return_value = mock_result

        # Call endpoint function directly (GET with list param doesn't route cleanly via TestClient)
        response = get_documents_by_ids_via_get(
            index_name="test_index", document_ids=["doc1"],
            marqo_config=get_config(), expose_facets=False
        )
        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 200)

    @mock.patch('marqo.tensor_search.tensor_search.get_documents_by_ids')
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


class ValidationApiTests(MarqoTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        test_index_request = cls.unstructured_marqo_index_request(schema_template_version=None)
        cls.indexes = cls.create_indexes([test_index_request])

        cls.test_index = cls.indexes[0]

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

    def test_apply_latest_schema_template_defaultDisabled(self):
        """
        Test that the apply_latest_schema_template endpoint returns 403 by default.
        """
        index_name = self.test_index.name
        response = self.client.post(f"/indexes/{index_name}/apply-latest-schema-template")
        self.assertEqual(response.status_code, 403)
        self.assertIn("This API endpoint is disabled", response.json()["detail"])

    def test_apply_latest_schema_template_disabled(self):
        """
        Test that the apply_latest_schema_template endpoint returns 403 when ops API is disabled explicitly.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'FALSE'}):
            index_name = self.test_index.name
            response = self.client.post(f"/indexes/{index_name}/apply-latest-schema-template")
            self.assertEqual(response.status_code, 403)
            self.assertIn("This API endpoint is disabled", response.json()["detail"])

    def test_apply_latest_schema_template_enabled(self):
        """
        Test that the apply_latest_schema_template endpoint is accessible when ops API is enabled.
        """
        with patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'TRUE'}):
            index_name = self.test_index.name
            response = self.client.post(f"/indexes/{index_name}/apply-latest-schema-template")
            self.assertEqual(response.status_code, 200)
            # Since the test index is created with schema_template_version defaulting to current version,
            # the shortcut is triggered
            current_version = version.get_version()
            self.assertEqual(f"Schema is already at current Marqo version {current_version}", response.json()["reason"])


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

        cls.add_documents(cls.config, AddDocsParams(
            index_name=cls.structured_index.name,
            docs=[{'field1': 'hello', 'field2': 'world'}],
        ))

        cls.add_documents(cls.config, AddDocsParams(
            index_name=cls.unstructured_index.name,
            docs=[{'field1': 'hello', 'field2': 'world'}],
            tensor_fields=['field1'],
        ))

    def test_search_timeout_short_timer_fails(self):
        # Set up the test API client with the correct env vars set
        with mock.patch.dict(os.environ, {
            "VESPA_SEARCH_TIMEOUT_MS": "1",
        }):
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
                for index in [self.unstructured_index, self.structured_index]:
                    with self.subTest(index=index.name):
                        res = self.client.post("/indexes/" + index.name + "/search?device=cpu", json={
                            "q": "irrelevant",
                            "searchMethod": "HYBRID"
                        })
                        # The search request must timeout, since the timeout is set to 1ms
                        self.assertEqual(res.status_code, 504)
                        self.assertEqual(res.json()["code"], "vector_store_timeout")
                        self.assertEqual(res.json()["type"], "invalid_request")

    def test_inference_cache_caches_query_string(self):
        with mock.patch.dict(os.environ, {
            "MARQO_API_INFERENCE_CACHE_SIZE": "10",
            "MARQO_API_INFERENCE_CACHE_TYPE": "LFU",
        }):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, CachingInference)
            with patch.object(inference.delegate, "vectorise", wraps=inference.delegate.vectorise) as mock_vectorise:
                with TestClient(api.app) as client:
                    for index in [self.unstructured_index, self.structured_index]:
                        with self.subTest(index=index.name):
                            mock_vectorise.reset_mock()
                            client.post("/indexes/" + index.name + "/search?telemetry=true",
                                        json={"q": f"hello {index.name}"})
                            mock_vectorise.assert_called_once()

                            # the second request with the same query should hit cache
                            mock_vectorise.reset_mock()
                            client.post("/indexes/" + index.name + "/search?telemetry=true",
                                        json={"q": f"hello {index.name}"})
                            mock_vectorise.assert_not_called()


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
        # Reload the api module to ensure env vars are re-read
        importlib.reload(sys.modules['marqo.tensor_search.api'])
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

    def test_approximate_threshold_with_lexical_method_error(self):
        """Test that approximateThreshold used with lexical method returns validation error"""
        response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
            "q": "test",
            "searchMethod": "LEXICAL",
            "approximateThreshold": 0.5
        })

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["type"], "value_error")
        self.assertIn("'approximateThreshold' is only valid for 'HYBRID' and 'TENSOR' search methods",
                      response.json()["detail"][0]["msg"])

    def test_approximate_threshold_with_approximate_false_error(self):
        """Test that approximateThreshold used when approximate=false returns validation error"""
        response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
            "q": "test",
            "searchMethod": "TENSOR",
            "approximate": False,
            "approximateThreshold": 0.5
        })

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["type"], "value_error")
        self.assertIn("'approximateThreshold' cannot be set when 'approximate' is False",
                      response.json()["detail"][0]["msg"])

    def test_approximate_threshold_outside_range_error(self):
        """Test that approximateThreshold outside of 0 to 1 range returns validation error"""
        invalid_thresholds = [-0.1, -1.0, 1.1, 2.0, 5.0]

        for threshold in invalid_thresholds:
            with self.subTest(threshold=threshold):
                response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
                    "q": "test",
                    "searchMethod": "TENSOR",
                    "approximateThreshold": threshold
                })

                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"][0]["type"], "value_error")
                self.assertIn(f"'approximateThreshold' must be between 0 and 1, got {threshold}",
                              response.json()["detail"][0]["msg"])

    def test_approximate_threshold_valid_values(self):
        """Test that valid approximateThreshold values work correctly with TENSOR search method"""
        valid_thresholds = [0.0, 0.5, 1.0]

        for threshold in valid_thresholds:
            with self.subTest(threshold=threshold):
                response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
                    "q": "test",
                    "searchMethod": "TENSOR",
                    "approximateThreshold": threshold
                })

                # Should return 200 (success) since these are valid combinations
                self.assertEqual(response.status_code, 200)
                self.assertIn("hits", response.json())

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

    def test_parse_request_object_should_parse_pydantic_v1_model(self):
        """Ensures parse_request_object parses pydantic v1 model"""

        class PydanticV1Model(pydantic.v1.BaseModel):
            field1: str

        request_obj_dict = {"field1": "hello"}

        model = api.parse_request_object(PydanticV1Model, request_obj_dict)

        self.assertEqual(model.field1, "hello")

    def test_parse_request_object_should_not_parse_pydantic_v2_model(self):
        """Ensures parse_request_object does not parse pydantic v2 model"""

        class PydanticV2Model(pydantic.BaseModel):
            field1: str

        request_obj_dict = {"field1": "hello"}

        with self.assertRaises(RuntimeError) as context:
            api.parse_request_object(PydanticV2Model, request_obj_dict)

        self.assertIn('no validator found for', str(context.exception))

    def test_parse_request_object_should_raise_request_validation_exception(self):
        """Ensures parse_request_object raises RequestValidationError on pydantic v1 validation error"""

        class PydanticV1Model(pydantic.v1.BaseModel):
            field2: str

        request_obj_dict = {"field1": "hello"}

        with self.assertRaises(RequestValidationError) as context:
            api.parse_request_object(PydanticV1Model, request_obj_dict)

        self.assertIn('field required', str(context.exception.errors()))

    def test_handle_pydantic_v1_validation_errors(self):
        """Test pydantic v1 ValidationError is correctly handled and converted to error response"""
        error = pydantic.v1.ValidationError(errors=[ErrorWrapper(ValueError("some message"), loc="doc")],
                                            model=SearchQuery)
        with patch("marqo.tensor_search.tensor_search.search", side_effect=error):
            response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
                "q": "test",
                "filter": ""
            })

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], InvalidArgError.code)
            self.assertEqual(response.json()["type"], InvalidArgError.error_type)
            assert "some message" in response.json()["message"]

    def test_handle_pydantic_v2_validation_errors(self):
        """Test pydantic v2 ValidationError is correctly handled and converted to error response"""
        error = pydantic.ValidationError.from_exception_data(
            title='SearchQuery',
            line_errors=[InitErrorDetails(
                type=PydanticCustomError('type1', 'some message'), loc=('doc',), input=...)]
        )
        with patch("marqo.tensor_search.tensor_search.search", side_effect=error):
            response = self.client.post("/indexes/" + self.structured_index.name + "/search?device=cpu", json={
                "q": "test",
                "filter": ""
            })

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], InvalidArgError.code)
            self.assertEqual(response.json()["type"], InvalidArgError.error_type)
            assert "some message" in response.json()["message"]

    # TODO: Test how marqo handles generic exceptions, including Exception, RunTimeError, ValueError, etc.


class TestUpdateIndexSettingsEndpoints(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_update_index_settings_enabled(self):
        """Test that update_index_settings endpoint works when ops API is enabled"""
        with mock.patch(
                "marqo.core.index_management.index_management.IndexManagement."
                "update_index_settings_by_settings_dict") as mock_update, \
                patch.dict('os.environ',  {EnvVars.MARQO_ENABLE_OPS_API: 'TRUE'}):

            mock_update.return_value = None

            response = self.client.patch(
                f"/indexes/index-1/index-settings",
                json={}
            )

            self.assertEqual(response.status_code, 200)
            mock_update.assert_called_once()

    def test_update_index_settings_disabled(self):
        """Test that update_index_settings endpoint works when ops API is enabled"""
        with mock.patch(
                "marqo.core.index_management.index_management.IndexManagement."
                "update_index_settings_by_settings_dict") as mock_update, \
                patch.dict('os.environ', {EnvVars.MARQO_ENABLE_OPS_API: 'False'}):
            mock_update.return_value = None

            response = self.client.patch(
                f"/indexes/index-1/index-settings",
                json={}
            )

            self.assertEqual(response.status_code, 403)
            mock_update.assert_not_called()


class TestModelApiEndpoints(MarqoTestCase):
    """Tests for model API endpoints"""

    def setUp(self):
        self.client = TestClient(api.app)

    def test_get_loaded_models_detailed(self):
        """Test get_loaded_models with detailed=True parameter"""
        with mock.patch('marqo.core.inference.model_manager_client.model_manager_client.ModelManagerClient.'
                         'get_loaded_models') as mock_get_model:

            mock_get_model.return_value = {
                "models": [
                    {
                        "modelName": "test-model",
                        "modelProperties": {"dimensions": 768}
                    }
                ]
            }

            response = self.client.get("/models?detailed=true")

            self.assertEqual(response.status_code, 200)
            mock_get_model.assert_called_once_with(True)

    def test_get_loaded_models_detailed_false(self):
        """Test get_loaded_models with detailed=false parameter"""
        with mock.patch('marqo.core.inference.model_manager_client.model_manager_client.ModelManagerClient.'
                         'get_loaded_models') as mock_get_model:

            mock_get_model.return_value = {
                "models": [
                    {
                        "modelName": "test-model",
                    }
                ]
            }

            response = self.client.get("/models?detailed=false")

            self.assertEqual(response.status_code, 200)
            mock_get_model.assert_called_once_with(False)

    def test_get_loaded_models_detailed_default(self):
        """Test get_loaded_models with default(false) parameter"""
        with mock.patch('marqo.core.inference.model_manager_client.model_manager_client.ModelManagerClient.'
                         'get_loaded_models') as mock_get_model:

            mock_get_model.return_value = {
                "models": [
                    {
                        "modelName": "test-model",
                    }
                ]
            }

            response = self.client.get("/models?")

            self.assertEqual(response.status_code, 200)
            mock_get_model.assert_called_once_with(False)

    def test_eject_model(self):
        """Test eject_model endpoint"""
        with mock.patch('marqo.core.inference.model_manager_client.model_manager_client.ModelManagerClient.'
                        'eject_model') as mock_eject_model:
            mock_eject_model.return_value = {"acknowledged": True}

            response = self.client.delete("/models?model_name=test-model")

            self.assertEqual(response.status_code, 200)
            mock_eject_model.assert_called_once_with(model_name="test-model")