import datetime
from copy import deepcopy
import marqo.tensor_search.delete_docs
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest, MqDeleteDocsResponse
from marqo.tensor_search import delete_docs
import marqo.tensor_search.tensor_search
from marqo.tensor_search import tensor_search
from marqo.errors import IndexNotFoundError
from tests.marqo_test import MarqoTestCase
import requests
from unittest.mock import patch
from marqo import errors
from marqo.tensor_search import enums
from tests.utils.transition import add_docs_caller
import os

class TestDeleteDocuments(MarqoTestCase):
    """module that has tests at the tensor_search level"""

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"

        self._delete_testing_indices()

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

    def _delete_testing_indices(self):
        for ix in [self.index_name_1]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_delete_documents(self):
        # first batch:
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder"},
            ], auto_refresh=True)
        count0_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"hooped": "absolutely ridic", "Darling": "A harbour in Sydney", "_id": "455"},
                {"efg": "hheeehehehhe", "_id": "at-at"}
            ], auto_refresh=True)
        count1_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        marqo.tensor_search.tensor_search.delete_documents(config=self.config, index_name=self.index_name_1, doc_ids=["455", "at-at"],
                                                           auto_refresh=True)
        count_post_delete = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        assert count_post_delete == count0_res

    def test_delete_docs_format(self):
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney", "_id": "1234"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder", "_id": "5678"},
            ], auto_refresh=True)

        res = marqo.tensor_search.tensor_search.delete_documents(config=self.config, doc_ids=["5678", "491"], index_name=self.index_name_1
                                                                 , auto_refresh=False)
        assert res["index_name"] == self.index_name_1
        assert res["type"] == "documentDeletion"
        assert res["status"] == "succeeded"
        assert res["details"]["receivedDocumentIds"] == 2
        assert res["details"]["deletedDocuments"] == 1
        assert "PT" in res["duration"]
        assert "Z" in res["startedAt"]
        assert "T" in res["finishedAt"]

    def test_only_specified_documents_are_deleted(self):
        # Add multiple documents
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"sample_field": "sample value", "_id": "unique_id_1"},
                {"sample_field": "sample value", "_id": "unique_id_2"},
                {"sample_field": "sample value", "_id": "unique_id_3"},
            ], auto_refresh=True
        )

        # Delete only one document
        tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["unique_id_1"], auto_refresh=True
        )

        # Try to retrieve the deleted document
        with self.assertRaises(errors.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="unique_id_1")

        # Check if the other documents still exist
        remaining_doc_1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1, document_id="unique_id_2")
        remaining_doc_2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1, document_id="unique_id_3")
        self.assertIsNotNone(remaining_doc_1)
        self.assertIsNotNone(remaining_doc_2)

    def test_delete_multiple_documents(self):
        # Create an index and add documents
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
                {"field1": "value2", "_id": "doc_id_2"},
                {"field1": "value3", "_id": "doc_id_3"},
            ],
            auto_refresh=True
        )

        # Delete multiple documents at once
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["doc_id_1", "doc_id_3"], auto_refresh=True
        )

        self.assertEqual(response["details"]["deletedDocuments"], 2)
        self.assertEqual(response["details"]["receivedDocumentIds"], 2)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.index_name_1)

        # Check if the remaining document is the correct one
        remaining_document = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1,
                                                              document_id="doc_id_2")
        self.assertEqual(remaining_document["_id"], "doc_id_2")

    def test_document_is_actually_deleted(self):
        # Add a document
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[{"sample_field": "sample value", "_id": "unique_id"}], auto_refresh=True
        )

        # Delete the document
        tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["unique_id"], auto_refresh=True
        )

        # Try to retrieve the deleted document
        with self.assertRaises(errors.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="unique_id")

    def test_multiple_documents_are_actually_deleted(self):
        # Add multiple documents
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"sample_field": "sample value", "_id": "unique_id_1"},
                {"sample_field": "sample value", "_id": "unique_id_2"},
            ], auto_refresh=True
        )

        # Delete the documents
        tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["unique_id_1", "unique_id_2"], auto_refresh=True
        )

        # Try to retrieve the deleted documents
        with self.assertRaises(errors.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1,
                                             document_id="unique_id_1")
        with self.assertRaises(errors.DocumentNotFoundError):
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1,
                                             document_id="unique_id_2")

    def test_delete_non_existent_document(self):
        # Attempt to delete a non-existent document
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["non_existent_id"], auto_refresh=True
        )

        self.assertEqual(response["details"]["deletedDocuments"], 0)
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.index_name_1)

    def test_delete_documents_from_non_existent_index(self):
        non_existent_index = "non-existent-index"

        with self.assertRaises(errors.IndexNotFoundError):
            tensor_search.delete_documents(
                config=self.config, index_name=non_existent_index, doc_ids=["unique_id_1"], auto_refresh=True
            )

    def test_delete_documents_with_empty_list(self):
        with self.assertRaises(errors.InvalidDocumentIdError):
            tensor_search.delete_documents(
                config=self.config, index_name=self.index_name_1, doc_ids=[], auto_refresh=True
            )

    def test_delete_documents_with_invalid_ids(self):
        with self.assertRaises(errors.InvalidDocumentIdError):
            tensor_search.delete_documents(
                config=self.config, index_name=self.index_name_1, doc_ids=[123, {"invalid": "id"}], auto_refresh=True
            )

    def test_delete_already_deleted_document(self):
        # Add a document
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
            ],
            auto_refresh=True
        )

        # Delete the document
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["doc_id_1"], auto_refresh=True
        )

        self.assertEqual(response["details"]["deletedDocuments"], 1)
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.index_name_1)

        # Attempt to delete the document again
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["doc_id_1"], auto_refresh=True
        )

        self.assertEqual(response["details"]["deletedDocuments"], 0)
        self.assertEqual(response["details"]["receivedDocumentIds"], 1)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.index_name_1)

    def test_delete_documents_mixed_valid_invalid_ids(self):
        # Add documents
        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"field1": "value1", "_id": "doc_id_1"},
                {"field1": "value2", "_id": "doc_id_2"},
            ],
            auto_refresh=True
        )

        # Delete documents using mixed valid and invalid document IDs
        response = tensor_search.delete_documents(
            config=self.config, index_name=self.index_name_1, doc_ids=["doc_id_1", "invalid_id"], auto_refresh=True
        )

        self.assertEqual(response["details"]["deletedDocuments"], 1)
        self.assertEqual(response["details"]["receivedDocumentIds"], 2)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["index_name"], self.index_name_1)

        # Check if the remaining document is the correct one
        remaining_document = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1,
                                                              document_id="doc_id_2")
        self.assertEqual(remaining_document["_id"], "doc_id_2")


class TestDeleteDocumentsEndpoint(MarqoTestCase):
    """Module that has tests at the tensor_search level"""

    def setUp(self) -> None:
        super().setUp()

        self.index_name_1 = "my-test-index-1"

        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_format_delete_docs_response(self):
        response = MqDeleteDocsResponse(
            index_name=self.index_name_1,
            status_string="succeeded",
            document_ids=["1", "2", "3"],
            deleted_docments_count=3,
            deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
            deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
        )

        formatted_response = delete_docs.format_delete_docs_response(response)

        self.assertEqual(formatted_response["index_name"], self.index_name_1)
        self.assertEqual(formatted_response["status"], "succeeded")
        self.assertEqual(formatted_response["type"], "documentDeletion")
        self.assertEqual(formatted_response["details"]["receivedDocumentIds"], 3)
        self.assertEqual(formatted_response["details"]["deletedDocuments"], 3)
        self.assertEqual(formatted_response["duration"], "PT5.0S")
        self.assertEqual(formatted_response["startedAt"], "2023-04-17T00:00:00Z")
        self.assertEqual(formatted_response["finishedAt"], "2023-04-17T00:00:05Z")

    def test_delete_documents_valid_request(self):
        config_copy = deepcopy(self.config)
        config_copy.backend = enums.SearchDb.opensearch
        request = MqDeleteDocsRequest(
            index_name=self.index_name_1, document_ids=["1", "2", "3"], auto_refresh=True
        )

        with patch("marqo.tensor_search.delete_docs.delete_documents_marqo_os") as mock_delete_documents_marqo_os:
            mock_delete_documents_marqo_os.return_value = MqDeleteDocsResponse(
                index_name=self.index_name_1,
                status_string="succeeded",
                document_ids=["1", "2", "3"],
                deleted_docments_count=3,
                deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
                deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
            )

            response = delete_docs.delete_documents(config_copy, request)

            mock_delete_documents_marqo_os.assert_called_once_with(config=config_copy, deletion_instruction=request)

    def test_delete_documents_empty_document_ids(self):
        config_copy = deepcopy(self.config)
        config_copy.backend = enums.SearchDb.opensearch
        request = MqDeleteDocsRequest(
            index_name=self.index_name_1, document_ids=[], auto_refresh=False
        )

        with self.assertRaises(errors.InvalidDocumentIdError):
            delete_docs.delete_documents(config_copy, request)

    def test_delete_documents_invalid_backend(self):
        config_copy = deepcopy(self.config)
        config_copy.backend = "unknown_backend"
        request = MqDeleteDocsRequest(
            index_name=self.index_name_1, document_ids=["1", "2", "3"], auto_refresh=False
        )

        with self.assertRaises(RuntimeError):
            delete_docs.delete_documents(config_copy, request)

    def test_delete_documents_marqo_os(self):
        config_copy = deepcopy(self.config)
        config_copy.backend = enums.SearchDb.opensearch
        request = MqDeleteDocsRequest(
            index_name=self.index_name_1, document_ids=["1", "2", "3"], auto_refresh=False
        )

        with patch("marqo.tensor_search.delete_docs.HttpRequests.post") as mock_post:
            mock_post.side_effect = [
                {
                    "items": [
                        {"delete": {"status": 200}},
                        {"delete": {"status": 200}},
                        {"delete": {"status": 200}},
                    ]
                }
            ]

            response = delete_docs.delete_documents_marqo_os(config_copy, request)

            self.assertEqual(response.index_name, self.index_name_1)
            self.assertEqual(response.status_string, "succeeded")
            self.assertEqual(response.document_ids, ["1", "2", "3"])
            self.assertEqual(response.deleted_docments_count, 3)

    def test_format_delete_docs_response_valid_input(self):
        mq_delete_res = MqDeleteDocsResponse(
            index_name="test_index",
            status_string="succeeded",
            document_ids=["1", "2", "3"],
            deleted_docments_count=3,
            deletion_start=datetime.datetime(2023, 4, 17, 0, 0, 0),
            deletion_end=datetime.datetime(2023, 4, 17, 0, 0, 5),
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
        }

        self.assertEqual(formatted_response, expected_response)

    def test_delete_documents_invalid_document_id(self):
        config_copy = deepcopy(self.config)
        config_copy.backend = enums.SearchDb.opensearch
        request = MqDeleteDocsRequest(
            index_name=self.index_name_1, document_ids=['hello', None, 123], auto_refresh=True
        )

        with self.assertRaises(errors.InvalidDocumentIdError):
            delete_docs.delete_documents(config_copy, request)

    def test_max_doc_delete_limit(self):
        max_delete_docs = 100
        mock_environ = {enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT: str(max_delete_docs)}

        doc_ids = [f"id_{x}" for x in range(max_delete_docs + 5)]

        @patch.dict(os.environ, mock_environ)
        def run():
            tensor_search.create_vector_index(
                index_name=self.index_name_1, index_settings={"index_defaults": {"model": 'random'}}, config=self.config)
            # over the limit:
            add_docs_caller(
                config=self.config, index_name=self.index_name_1, docs=[
                    {"_id": x, 'Bad field': "blh "} for x in doc_ids
                ],
                auto_refresh=True, update_mode='update')
            try:
                tensor_search.delete_documents(config=self.config, index_name=self.index_name_1, doc_ids=doc_ids, auto_refresh=True)
                raise AssertionError
            except errors.InvalidArgError as e:
                pass
            # under the limit
            update_res = tensor_search.delete_documents(
                config=self.config, index_name=self.index_name_1, doc_ids=doc_ids[:90],
                auto_refresh=True)
            assert update_res['details']['receivedDocumentIds'] == update_res['details']['deletedDocuments']
            assert update_res['details']['receivedDocumentIds'] == 90
            return True

        assert run()

    def test_max_doc_delete_default_limit(self):
        default_limit = 10000

        @patch.dict(os.environ, dict())
        def run():
            assert default_limit == tensor_search.utils.read_env_vars_and_defaults_ints(
                enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT)
            return True

        assert run()
