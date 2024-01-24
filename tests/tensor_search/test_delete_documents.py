import datetime
import unittest
from copy import deepcopy, copy
import marqo.tensor_search.delete_docs
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest, MqDeleteDocsResponse
from marqo.tensor_search import delete_docs
import marqo.tensor_search.tensor_search
from marqo.core.index_management import index_management
from marqo.core.models.marqo_index import Model
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
import requests
from unittest.mock import patch
from marqo.core import exceptions as core_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.tensor_search import enums
from tests.utils.transition import add_docs_caller, add_docs_batched
import os
from marqo.vespa.models.delete_document_response import DeleteBatchDocumentResponse, DeleteBatchResponse

class TestDeleteDocuments(MarqoTestCase):
    """module that has tests at the tensor_search level"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        text_index_with_random_model_request = cls.unstructured_marqo_index_request(model=Model(name='random'))

        cls.create_indexes([
            text_index_with_random_model_request
        ])

        cls.text_index_with_random_model = text_index_with_random_model_request.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_delete_documents(self):
        # first batch:
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder"},
            ], tensor_fields=["f1"])

        count0_res = self.monitoring.get_index_stats_by_name(self.text_index_with_random_model).number_of_documents

        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"hooped": "absolutely ridic", "Darling": "A harbour in Sydney", "_id": "455"},
                {"efg": "hheeehehehhe", "_id": "at-at"}
            ], tensor_fields=["f1"])

        count1_res = self.monitoring.get_index_stats_by_name(self.text_index_with_random_model).number_of_documents

        marqo.tensor_search.tensor_search.delete_documents(config=self.config, index_name=self.text_index_with_random_model, doc_ids=["455", "at-at"])

        count_post_delete = self.monitoring.get_index_stats_by_name(self.text_index_with_random_model).number_of_documents

        self.assertEqual(count_post_delete, count0_res)
        self.assertEqual(count1_res, count0_res + 2)

    def test_delete_docs_format(self):
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney", "_id": "1234"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder", "_id": "5678"},
            ], tensor_fields=["f1"])

        res = marqo.tensor_search.tensor_search.delete_documents(config=self.config, doc_ids=["5678", "491"], index_name=self.text_index_with_random_model)
        self.assertEqual(res["index_name"], self.text_index_with_random_model)
        self.assertEqual(res["type"], "documentDeletion")
        self.assertEqual(res["status"], "succeeded")
        self.assertEqual(res["details"]["receivedDocumentIds"], 2)
        self.assertEqual(res["details"]["deletedDocuments"], 2)  # 491 is counted in deletedDocuments count, even if it doesn't exist
        assert "PT" in res["duration"]
        assert "Z" in res["startedAt"]
        assert "T" in res["finishedAt"]

    def test_only_specified_documents_are_deleted(self):
        # Add multiple documents
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"sample_field": "sample value", "_id": "unique_id_1"},
                {"sample_field": "sample value", "_id": "unique_id_2"},
                {"sample_field": "sample value", "_id": "unique_id_3"},
            ], tensor_fields=[]
        )

        # Delete only one document
        tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["unique_id_1"],
        )

        # Try to retrieve the deleted document
        with self.assertRaises(api_exceptions.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model, document_id="unique_id_1")

        # Check if the other documents still exist
        remaining_doc_1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.text_index_with_random_model, document_id="unique_id_2")
        remaining_doc_2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.text_index_with_random_model, document_id="unique_id_3")
        self.assertIsNotNone(remaining_doc_1)
        self.assertIsNotNone(remaining_doc_2)

    def test_delete_multiple_documents(self):
        # Create an index and add documents
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
                {"field1": "value2", "_id": "doc_id_2"},
                {"field1": "value3", "_id": "doc_id_3"},
            ],
            tensor_fields=[]
        )

        # Delete multiple documents at once
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["doc_id_1", "doc_id_3"],
        )

        self.assertEqual(response["details"]["deletedDocuments"], 2)
        self.assertEqual(response["details"]["receivedDocumentIds"], 2)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.text_index_with_random_model)

        # Check if the remaining document is the correct one
        remaining_document = tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model,
                                                              document_id="doc_id_2")
        self.assertEqual(remaining_document["_id"], "doc_id_2")

    def test_document_is_actually_deleted(self):
        # Add a document
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[{"sample_field": "sample value", "_id": "unique_id"}], tensor_fields=[]
        )

        # Delete the document
        tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["unique_id"],
        )

        # Try to retrieve the deleted document
        with self.assertRaises(api_exceptions.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model, document_id="unique_id")

    def test_multiple_documents_are_actually_deleted(self):
        # Add multiple documents
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"sample_field": "sample value", "_id": "unique_id_1"},
                {"sample_field": "sample value", "_id": "unique_id_2"},
            ], tensor_fields=[]
        )

        # Delete the documents
        tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["unique_id_1", "unique_id_2"],
        )

        # Try to retrieve the deleted documents
        with self.assertRaises(api_exceptions.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model,
                                             document_id="unique_id_1")
        with self.assertRaises(api_exceptions.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model,
                                             document_id="unique_id_2")

    def test_delete_non_existent_document(self):
        # Attempt to delete a non-existent document
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["non_existent_id"],
        )

        self.assertEqual(response["details"]["deletedDocuments"], 1)    # Vespa still returns a 200, so the doc is counted as deleted
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.text_index_with_random_model)

    def test_delete_documents_from_non_existent_index(self):
        non_existent_index = "non-existent-index"

        with self.assertRaises(api_exceptions.IndexNotFoundError):
            tensor_search.delete_documents(
                config=self.config, index_name=non_existent_index, doc_ids=["unique_id_1"],
            )

    def test_delete_documents_with_empty_list(self):
        with self.assertRaises(api_exceptions.InvalidDocumentIdError):
            tensor_search.delete_documents(
                config=self.config, index_name=self.text_index_with_random_model, doc_ids=[],
            )

    def test_delete_documents_with_invalid_ids(self):
        with self.assertRaises(api_exceptions.InvalidDocumentIdError):
            tensor_search.delete_documents(
                config=self.config, index_name=self.text_index_with_random_model, doc_ids=[123, {"invalid": "id"}],
            )

    def test_delete_already_deleted_document(self):
        # Add a document
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
            ],
            tensor_fields=[]
        )

        # Delete the document
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["doc_id_1"],
        )

        self.assertEqual(response["details"]["deletedDocuments"], 1)
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.text_index_with_random_model)

        # Attempt to delete the document again
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["doc_id_1"],
        )

        self.assertEqual(response["details"]["deletedDocuments"], 1)    # Vespa still returns a 200, so the doc is counted as deleted
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.text_index_with_random_model)

    def test_delete_documents_mixed_valid_invalid_ids(self):
        # Add documents
        add_docs_caller(
            config=self.config, index_name=self.text_index_with_random_model,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
                {"field1": "value2", "_id": "doc_id_2"},
            ],
            tensor_fields=[]
        )

        # Delete documents using mixed valid and invalid document IDs
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.text_index_with_random_model, doc_ids=["doc_id_1", "invalid_id"],
        )

        self.assertEqual(response["details"]["deletedDocuments"], 2)    # Vespa still returns a 200, so the doc is counted as deleted
        self.assertEqual(response["details"]["receivedDocumentIds"], 2)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.text_index_with_random_model)

        # Check if the remaining document is the correct one
        remaining_document = tensor_search.get_document_by_id(config=self.config, index_name=self.text_index_with_random_model,
                                                              document_id="doc_id_2")
        self.assertEqual(remaining_document["_id"], "doc_id_2")


@unittest.skip
class TestDeleteDocumentsEndpoint(MarqoTestCase):
    """Module that has tests at the tensor_search level"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        text_index_with_random_model = cls.unstructured_marqo_index_request(model=Model(name='random'))

        cls.create_indexes([
            text_index_with_random_model
        ])

        cls.text_index_with_random_model = text_index_with_random_model.name

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.mock_result_list = [
            {
                "delete": {
                    '_id': '1',
                    'result': 'deleted',
                    'status': 200
                }
            },
            {
                "delete": {
                    '_id': '2',
                    'result': 'deleted',
                    'status': 200
                }
            },
            {
                "delete": {
                    '_id': '3',
                    'result': 'deleted',
                    'status': 200
                }
            },
        ]
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()


    def test_format_delete_docs_response(self):
        response = MqDeleteDocsResponse(
            index_name=self.text_index_with_random_model,
            status_string="succeeded",
            document_ids=["1", "2", "3"],
            result_list=self.mock_result_list,
            deleted_documents_count=3,
            deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
            deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
        )

        formatted_response = delete_docs.format_delete_docs_response(response)

        self.assertEqual(formatted_response["index_name"], self.text_index_with_random_model)
        self.assertEqual(formatted_response["status"], "succeeded")
        self.assertEqual(formatted_response["type"], "documentDeletion")
        self.assertEqual(formatted_response["details"]["receivedDocumentIds"], 3)
        self.assertEqual(formatted_response["details"]["deletedDocuments"], 3)
        self.assertEqual(formatted_response["duration"], "PT5.0S")
        self.assertEqual(formatted_response["startedAt"], "2023-04-17T00:00:00Z")
        self.assertEqual(formatted_response["finishedAt"], "2023-04-17T00:00:05Z")
        self.assertEqual(formatted_response["items"], self.mock_result_list)

    def test_delete_documents_valid_request(self):
        config_copy = copy(self.config)
        config_copy.backend = enums.SearchDb.vespa
        request = MqDeleteDocsRequest(
            index_name=self.text_index_with_random_model, document_ids=["1", "2", "3"]
        )

        with patch("marqo.tensor_search.delete_docs.delete_documents_vespa") as mock_delete_documents_vespa:
            mock_delete_documents_vespa.return_value = MqDeleteDocsResponse(
                index_name=self.text_index_with_random_model,
                status_string="succeeded",
                document_ids=["1", "2", "3"],
                result_list=self.mock_result_list,
                deleted_documents_count=3,
                deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
                deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
            )

            response = delete_docs.delete_documents(config_copy, request)

            mock_delete_documents_vespa.assert_called_once_with(config=config_copy, deletion_instruction=request)

    def test_delete_documents_empty_document_ids(self):
        config_copy = copy(self.config)
        config_copy.backend = enums.SearchDb.vespa
        request = MqDeleteDocsRequest(
            index_name=self.text_index_with_random_model, document_ids=[]
        )

        with self.assertRaises(api_exceptions.InvalidDocumentIdError):
            delete_docs.delete_documents(config_copy, request)

    def test_delete_documents_invalid_backend(self):
        config_copy = copy(self.config)
        config_copy.backend = "unknown_backend"
        request = MqDeleteDocsRequest(
            index_name=self.text_index_with_random_model, document_ids=["1", "2", "3"]
        )

        with self.assertRaises(RuntimeError):
            delete_docs.delete_documents(config_copy, request)

    def test_delete_documents_vespa(self):
        config_copy = copy(self.config)
        config_copy.backend = enums.SearchDb.vespa
        request = MqDeleteDocsRequest(
            index_name=self.text_index_with_random_model, document_ids=["1", "2", "3"]
        )

        with patch("marqo.vespa.vespa_client.VespaClient.delete_batch") as mock_delete_batch:
            mock_delete_batch.return_value = DeleteBatchResponse(
                responses=[
                    DeleteBatchDocumentResponse(status=200, pathId='/document/v1/mytestindex1/mytestindex1/docid/1',
                                                id='id:mytestindex1:mytestindex1::1', message=None),
                    DeleteBatchDocumentResponse(status=200, pathId='/document/v1/mytestindex1/mytestindex1/docid/2',
                                                id='id:mytestindex1:mytestindex1::2', message=None),
                    DeleteBatchDocumentResponse(status=200, pathId='/document/v1/mytestindex1/mytestindex1/docid/3',
                                                id='id:mytestindex1:mytestindex1::3', message=None)
                ],
                errors=False
            )

            response = delete_docs.delete_documents_vespa(config_copy, request)

            self.assertEqual(response.index_name, self.text_index_with_random_model)
            self.assertEqual(response.status_string, "succeeded")
            self.assertEqual(response.document_ids, ["1", "2", "3"])
            self.assertEqual(response.deleted_documents_count, 3)
            self.assertEqual(response.result_list, [item["delete"] for item in self.mock_result_list])

    def test_format_delete_docs_response_valid_input(self):
        mq_delete_res = MqDeleteDocsResponse(
            index_name="test_index",
            status_string="succeeded",
            document_ids=["1", "2", "3"],
            deleted_documents_count=3,
            deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
            deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
            result_list=self.mock_result_list
        )

        formatted_response = delete_docs.format_delete_docs_response(mq_delete_res)
        expected_response = {
            "index_name": "test_index",
            "status": "succeeded",
            "type": "documentDeletion",
            "details": {
                "receivedDocumentIds": 3,
                "deletedDocuments": 3,
            },
            "duration": "PT5.0S",
            "startedAt": "2023-04-17T00:00:00Z",
            "finishedAt": "2023-04-17T00:00:05Z",
            "items": self.mock_result_list
        }

        self.assertEqual(formatted_response, expected_response)

    def test_delete_documents_invalid_document_id(self):
        config_copy = copy(self.config)
        config_copy.backend = enums.SearchDb.vespa
        request = MqDeleteDocsRequest(
            index_name=self.text_index_with_random_model, document_ids=['hello', None, 123]
        )

        with self.assertRaises(api_exceptions.InvalidDocumentIdError):
            delete_docs.delete_documents(config_copy, request)

    def test_max_doc_delete_limit(self):
        max_delete_docs = 100
        mock_environ = {enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT: str(max_delete_docs)}

        doc_ids = [f"id_{x}" for x in range(max_delete_docs + 5)]

        @patch.dict(os.environ, mock_environ)
        def run():
            # over the limit:
            docs = [{"_id": x, 'Bad field': "blh "} for x in doc_ids]

            add_docs_batched(
                config=self.config, index_name=self.text_index_with_random_model,
                docs=docs, auto_refresh=False, device="cpu", tensor_fields=[]
            )
            try:
                tensor_search.delete_documents(config=self.config, index_name=self.text_index_with_random_model, doc_ids=doc_ids)
                raise AssertionError
            except api_exceptions.InvalidArgError as e:
                pass
            # under the limit
            update_res = tensor_search.delete_documents(
                config=self.config, index_name=self.text_index_with_random_model, doc_ids=doc_ids[:90])
            self.assertEqual(update_res['details']['receivedDocumentIds'], update_res['details']['deletedDocuments'])
            self.assertEqual(update_res['details']['receivedDocumentIds'], 90)
            return True

        assert run()

    def test_max_doc_delete_default_limit(self):
        default_limit = 10000

        @patch.dict(os.environ, dict())
        def run():
            self.assertEqual(default_limit, tensor_search.utils.read_env_vars_and_defaults_ints(
                enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT)
            )
            return True

        assert run()

