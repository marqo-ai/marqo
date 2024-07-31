from tests.marqo_test import MarqoTestCase


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
            (123, 500, "Marqo vector store returns an unexpected error with this document")
        ]
        for status, expected_status, expected_message in test_cases:
            with self.subTest(status=status):
                result_status, result_message = self.document.translate_vespa_document_response(status)
                self.assertEqual(result_status, expected_status)
                self.assertEqual(result_message, expected_message)