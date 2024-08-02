from tests.marqo_test import MarqoTestCase
from unittest.mock import patch


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