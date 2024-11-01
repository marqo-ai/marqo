from marqo.core.models.marqo_add_documents_response import (MarqoAddDocumentsResponse, MarqoAddDocumentsItem,
                                                            MarqoBaseDocumentsResponse)
from marqo.core.models.marqo_get_documents_by_id_response import (MarqoGetDocumentsByIdsResponse,
                                                                  MarqoGetDocumentsByIdsItem)
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from tests.marqo_test import MarqoTestCase


class TestDocumentsCommon(MarqoTestCase):
    """A test suite for common methods in documents module."""

    def setUp(self):
        super().setUp()
        self.document = self.config.document

    def test_marqoAddDocumentsResponseDictFormat(self):
        """A test to verify that the MarqoAddDocumentsResponse is serialised correctly."""
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
        """A test to verify that the MarqoUpdateDocumentsResponse is serialised correctly."""
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
        """A test to verify that the MarqoGetDocumentsByIdsResponse is serialised correctly."""
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

    def test_marqo_base_documents_response_dictExcludesBatchResponseStats(self):
        """A test to verify that _batch_response_stats is excluded from the response."""
        response = MarqoBaseDocumentsResponse()
        result = response.dict()
        self.assertNotIn('_batch_response_stats', result)

    def test_marqo_base_documents_response_dictExcludesCustomFields(self):
        """A test to verify that custom fields are excluded from the response."""
        class CustomResponse(MarqoBaseDocumentsResponse):
            custom_field: str = "value"

        response = CustomResponse()
        result = response.dict(exclude={'custom_field'})
        self.assertNotIn('custom_field', result)
        self.assertNotIn('_batch_response_stats', result)

    def test_marqo_base_documents_response_dictTypeErrorWhenExcludeIsNotSet(self):
        """A test to verify that a TypeError is raised when exclude is not a set."""
        response = MarqoBaseDocumentsResponse()
        with self.assertRaises(TypeError):
            response.dict(
                exclude=['_batch_response_stats']
            )  # This should raise a TypeError because exclude must be a set

    def test_marqo_base_documents_response_dictIncludesOtherFields(self):
        """A test to verify that other fields are included in the response."""
        class CustomResponse(MarqoBaseDocumentsResponse):
            custom_field: str = "value"

        response = CustomResponse()
        result = response.dict()
        self.assertIn('custom_field', result)
        self.assertNotIn('_batch_response_stats', result)