from marqo.api.models.get_batch_documents_request import GetBatchDocumentsRequest
from unittest import TestCase
from pydantic.v1 import ValidationError


class TestGetBatchDocumentsRequestModel(TestCase):

    def test_valid_get_batch_documents_request_model(self):
        test_cases = [
            (
                ["id1"], ["id1"], "valid input with 1 id"
            ),
            (
                ["id1", "id2"], ["id1", "id2"], "valid input with 2 ids"
            ),
            (
                [f"id{i}" for i in range(1000)], [f"id{i}" for i in range(1000)], "valid input with 1000 ids"
            ),
            (
                ["id1", 1], ["id1", "1"], "A valid one as Pydantic will coerce the int to str"
            ),
            (
                {"id1", }, ["id1"], "A valid one as Pydantic will coerce the set to list"
            )
        ]

        for document_ids, expected_document_ids, msg in test_cases:
            with self.subTest(msg=msg):
                request_model = GetBatchDocumentsRequest(document_ids=document_ids)
                self.assertEqual(expected_document_ids, request_model.document_ids)

                aliased_model = GetBatchDocumentsRequest(**{"documentIds": document_ids})
                self.assertEqual(expected_document_ids, aliased_model.document_ids)

    def test_invalid_get_batch_documents_request_model(self):
        test_cases = [
            (
                [], "empty list"
            ),
            (
                [{"ids:": "1"}], "list of dicts"
            ),
            (
                None, "None instead of list"
            )
        ]
        for document_ids, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValidationError):
                    GetBatchDocumentsRequest(document_ids=document_ids)