from tests.marqo_test import MarqoTestCase
from unittest.mock import patch
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_get_documents_by_id_response import (MarqoGetDocumentsByIdsResponse,
                                                                  MarqoGetDocumentsByIdsItem)
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem


class TestDocumentsCommon(MarqoTestCase):
    """A test suite for common methods in documents module."""

    def setUp(self):
        super().setUp()
        self.document = self.config.document

    def test_translate_vespa_document_response_status(self):
        test_cases = [
            (200, 200, None),
            (404, 404, "Document does not exist in the index"),
            (412, 404, "Document does not exist in the index"),
            (429, 429, "Marqo vector store receives too many requests. Please try again later"),
            (507, 400, "Marqo vector store is out of memory or disk space"),
            (123, 500, "Marqo vector store returns an unexpected error with this document"),
            (400, 500, "Marqo vector store returns an unexpected error with this document"),
            # generic 400 error without specific message
            (400, 400, "The document contains invalid characters in the fields. Original error: could not parse field"),
            # specific 400 error
        ]
        for status, expected_status, expected_message in test_cases:
            with self.subTest(status=status):
                if status == 400 and "could not parse field" in expected_message:
                    result_status, result_message = self.document.translate_vespa_document_response(
                        status,
                        "could not parse field"
                    )
                else:
                    result_status, result_message = self.document.translate_vespa_document_response(status)
                self.assertEqual(result_status, expected_status)
                if expected_message:
                    self.assertIn(expected_message, result_message)

    def test_translate_vespa_document_response_logging(self):
        with patch("marqo.core.document.document.logger.error") as mock_log_error:
            status = 400
            self.document.translate_vespa_document_response(status)
        mock_log_error.assert_called_once()

    def test_marqoAddDocumentsResponseDictFormat(self):
        item = MarqoAddDocumentsItem(status=200, id="id", message="message", error="error", code="code")
        res = MarqoAddDocumentsResponse(errors=False, processingTimeMs=0.0, index_name="index_name", items=[item])

        returned_dict = res.dict(exclude_none=True, by_alias=True)
        expected_dict = {
            "errors": False,
            "processingTimeMs": 0.0,
            "index_name": "index_name",
            "items": [
                {
                    "status": 200,
                    "_id": "id",
                    "message": "message",
                    "error": "error",
                    "code": "code"
                }
            ]
        }
        self.assertEqual(expected_dict, returned_dict)

    def test_marqoUpdateDocumentsResponseDictFormat(self):
        item = MarqoUpdateDocumentsItem(id="id", status=200, message="message", error="error")
        res = MarqoUpdateDocumentsResponse(errors=False, index_name="index_name", items=[item], processingTimeMs=0.0)

        returned_dict = res.dict(exclude_none=True, by_alias=True)
        expected_dict = {
            "errors": False,
            "index_name": "index_name",
            "items": [
                {
                    "_id": "id",
                    "status": 200,
                    "message": "message",
                    "error": "error"
                }
            ],
            "processingTimeMs": 0.0
        }
        self.assertEqual(expected_dict, returned_dict)

    def test_marqoGetDocumentsByIdsResponseDictFormat(self):
        item = MarqoGetDocumentsByIdsItem(id="id", status=200, message="message", found=True)
        res = MarqoGetDocumentsByIdsResponse(errors=False, results=[item])

        returned_dict = res.dict(exclude_none=True, by_alias=True)
        expected_dict = {
            "errors": False,
            "results": [
                {
                    "_id": "id",
                    "status": 200,
                    "message": "message",
                    "_found": True
                }
            ]
        }
        self.assertEqual(expected_dict, returned_dict)