import unittest
from unittest.mock import MagicMock, patch, Mock

from marqo.core.document.document import Document
from marqo.core.exceptions import InternalError
from marqo.core.models.marqo_index import IndexType
from marqo.core.semi_structured_vespa_index.common import (
    VESPA_FIELD_ID, INT_FIELDS, FLOAT_FIELDS, VESPA_DOC_FIELD_TYPES, VESPA_DOC_VERSION_UUID
)
from marqo.vespa.models.get_document_response import GetBatchDocumentResponse, GetBatchResponse
from marqo.vespa.models import UpdateDocumentsBatchResponse, VespaDocument
from tests.unit_tests.marqo_test import MarqoTestCase


class TestDocumentPartialUpdateMapHandling(MarqoTestCase):
    """Unit tests for partial_update_documents map handling in Document class.

    Covers the code path where semi-structured indexes with map fields
    fetch existing documents from Vespa via get_batch."""

    def setUp(self):
        super().setUp()
        self.mock_vespa_client = MagicMock()
        self.mock_index_management = MagicMock()
        self.mock_inference = MagicMock()
        self.document = Document(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference,
        )

        self.semi_structured_index = self.semi_structured_marqo_index(
            name="test_index",
            schema_name="test_schema",
            marqo_version="2.16.0",
        )

    def test_nonexistent_map_doc_returns_error_without_crash(self):
        """Test that a non-existent doc with map fields returns a per-document error.

        Covers lines 158-159 (fetch_ids, get_batch call),
        169 (enumerate loop), and 177-178 (fetch_ids[idx] lookup and error append)."""
        docs = [
            {"_id": "missing-doc#special", "int_map": {"k": 1}},
            {"_id": "valid-doc", "some_field": 42},
        ]

        # get_batch returns 404 for missing doc
        missing_resp = GetBatchDocumentResponse(
            status=404,
            pathId="/document/v1/test_schema/test_schema/docid/missing-doc",
            id="id:test_schema:test_schema::missing-doc",
            message="Document not found",
        )
        self.mock_vespa_client.get_batch.return_value = GetBatchResponse(
            responses=[missing_resp], errors=True
        )

        # update_documents_batch for the valid doc
        mock_update_resp = MagicMock(spec=UpdateDocumentsBatchResponse)
        mock_update_resp.errors = False
        mock_vespa_item = MagicMock()
        mock_vespa_item.status = 200
        mock_vespa_item.id = "id:test_schema:test_schema::valid-doc"
        mock_update_resp.responses = [mock_vespa_item]
        self.mock_vespa_client.update_documents_batch.return_value = mock_update_resp
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, None)

        with patch('marqo.core.document.document.vespa_index_factory') as mock_factory:
            mock_vespa_index = MagicMock()
            mock_vespa_index.to_vespa_partial_document.return_value = {"id": "valid-doc", "fields": {}}
            mock_vespa_index.get_vespa_id_field.return_value = "id"
            mock_factory.return_value = mock_vespa_index

            result = self.document.partial_update_documents(docs, self.semi_structured_index)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.items), 2)

        # First doc: error with original ID preserved (including #)
        self.assertEqual(400, result.items[0].status)
        self.assertEqual("missing-doc#special", result.items[0].id)
        self.assertIsNotNone(result.items[0].error)

        # Second doc: success
        self.assertEqual(200, result.items[1].status)

    def test_response_count_mismatch_raises_internal_error(self):
        """Test that a mismatch between fetch_ids and responses raises InternalError.

        Covers lines 163-164 (response count validation)."""
        docs = [
            {"_id": "doc1", "int_map": {"k": 1}},
            {"_id": "doc2", "int_map": {"k": 2}},
        ]

        # Return only 1 response for 2 requested IDs
        resp = GetBatchDocumentResponse(
            status=200,
            pathId="/document/v1/test_schema/test_schema/docid/doc1",
            id="id:test_schema:test_schema::doc1",
            fields={VESPA_FIELD_ID: "doc1"},
        )
        self.mock_vespa_client.get_batch.return_value = GetBatchResponse(
            responses=[resp], errors=False
        )

        with patch('marqo.core.document.document.vespa_index_factory') as mock_factory:
            mock_factory.return_value = MagicMock()
            with self.assertRaises(InternalError) as ctx:
                self.document.partial_update_documents(docs, self.semi_structured_index)

        self.assertIn("does not match", str(ctx.exception))

    def test_existing_map_doc_passes_fetched_data_to_vespa_index(self):
        """Test that when get_batch finds a document, its fields are passed as
        existing_vespa_document to to_vespa_partial_document for map merging."""
        docs = [
            {"_id": "existing-doc", "int_map": {"k": 1}},
        ]

        fetched_fields = {
            VESPA_FIELD_ID: "existing-doc",
            INT_FIELDS: '{"k": 5}',
            FLOAT_FIELDS: "{}",
            VESPA_DOC_FIELD_TYPES: "{}",
            VESPA_DOC_VERSION_UUID: "abc",
        }
        existing_resp = GetBatchDocumentResponse(
            status=200,
            pathId="/document/v1/test_schema/test_schema/docid/existing-doc",
            id="id:test_schema:test_schema::existing-doc",
            fields=fetched_fields,
        )
        self.mock_vespa_client.get_batch.return_value = GetBatchResponse(
            responses=[existing_resp], errors=False
        )

        mock_update_resp = MagicMock(spec=UpdateDocumentsBatchResponse)
        mock_update_resp.errors = False
        mock_update_resp.responses = []
        self.mock_vespa_client.update_documents_batch.return_value = mock_update_resp

        with patch('marqo.core.document.document.vespa_index_factory') as mock_factory:
            mock_vespa_index = MagicMock()
            mock_vespa_index.to_vespa_partial_document.return_value = {"id": "existing-doc", "fields": {}}
            mock_vespa_index.get_vespa_id_field.return_value = "id"
            mock_factory.return_value = mock_vespa_index

            self.document.partial_update_documents(docs, self.semi_structured_index)

        # Verify to_vespa_partial_document received the fetched Vespa document
        # (second arg is the existing doc used for map merging)
        mock_vespa_index.to_vespa_partial_document.assert_called_once()
        call_args = mock_vespa_index.to_vespa_partial_document.call_args[0]
        existing_doc_arg = call_args[1]
        self.assertIsNotNone(existing_doc_arg,
                             "Existing vespa document should be passed for map merging")
        # document.dict() nests fields under 'fields' key
        self.assertEqual(existing_doc_arg["fields"][VESPA_FIELD_ID], "existing-doc")
        self.assertEqual(existing_doc_arg["fields"][INT_FIELDS], '{"k": 5}')

    def test_vespa_error_message_is_surfaced_in_response(self):
        """Test that Vespa's error message is preserved in the response item."""
        docs = [
            {"_id": "bad-doc", "int_map": {"k": 1}},
        ]

        error_resp = GetBatchDocumentResponse(
            status=400,
            pathId="",
            message="Invalid document ID: bad-doc. Original error: some error",
        )
        self.mock_vespa_client.get_batch.return_value = GetBatchResponse(
            responses=[error_resp], errors=True
        )

        mock_update_resp = MagicMock(spec=UpdateDocumentsBatchResponse)
        mock_update_resp.errors = False
        mock_update_resp.responses = []
        self.mock_vespa_client.update_documents_batch.return_value = mock_update_resp

        with patch('marqo.core.document.document.vespa_index_factory') as mock_factory:
            mock_factory.return_value = MagicMock()
            mock_factory.return_value.get_vespa_id_field.return_value = "id"

            result = self.document.partial_update_documents(docs, self.semi_structured_index)

        self.assertTrue(result.errors)
        self.assertEqual(400, result.items[0].status)
        # The vespa message should be surfaced, not the generic one
        self.assertIn("Invalid document ID", result.items[0].error)
        self.assertIn("Invalid document ID", result.items[0].message)


if __name__ == '__main__':
    unittest.main()
