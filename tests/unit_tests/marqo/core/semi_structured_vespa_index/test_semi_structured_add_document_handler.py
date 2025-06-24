from unittest.mock import Mock, patch, MagicMock
import numpy as np
from marqo.core.exceptions import AddDocumentsError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import (
    SemiStructuredAddDocumentsHandler,
    SemiStructuredFieldCountConfig
)
from marqo.core.inference.api import InferenceRequest, InferenceResult, InferenceErrorModel, Modality
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.feed_response import FeedBatchResponse, FeedBatchDocumentResponse
from tests.unit_tests.marqo_test import MarqoTestCase


class TestSemiStructuredAddDocumentsHandler(MarqoTestCase):

    def setUp(self):
        self.mock_vespa_client = Mock()
        self.mock_index_management = Mock()
        self.mock_inference = Mock()
        self.field_count_config = SemiStructuredFieldCountConfig(
            max_tensor_field_count=10,
            max_lexical_field_count=10,
            max_string_array_field_count=10
        )

        # Setup basic inference mock
        def vectorise_side_effect(request: InferenceRequest) -> InferenceResult:
            result = []
            for content in request.contents:
                result.append([('chunk1', np.array([1.0, 2.0]))])
            return InferenceResult(result=result)

        self.mock_inference.vectorise.side_effect = vectorise_side_effect

    @patch('marqo.core.inference.modality_utils.infer_modality')
    def test_add_documents_success(self, mock_infer_modality):
        """Test document addition with variety of field types and language mappings"""
        # Cover different field types
        docs = [
            {
                "_id": "doc1",
                "title": "Hola mundo",
                "description": "Este es un documento en español",
                "content": "Machine learning content for vectorization",
                "image_url": "https://example.com/image1.jpg",
                "categories": ["tecnología", "ciencia"],
                "tags": ["AI", "ML", "NLP"],
                "price": 99.99,
                "available": True,
                "rating": 4,
                "my_custom_vector": {"vector": [0.1] * 384}
            },
            {
                "_id": "doc2",
                "title": "Hello world",
                "description": "This is a document in English",
                "content": "Natural language processing examples",
                "categories": ["technology", "science"],
                "tags": ["deep learning", "neural networks"],
                "price": 149.50,
                "available": False,
                "rating": 5,
                "multimodal_content": "Text with potential image references",
                "audio_description": "Voice content for audio processing",
                "combined_content": "Combined text and image content"
            },
            {
                "_id": "doc3",
                "title": "Computer vision introduction",
                "description": "A comprehensive guide to computer vision",
                "content": "Computer vision and image processing",
                "categories": ["IT", "research"],
                "tags": ["research", "computer vision"],
                "embedding_vector": {"vector": [0.5] * 384}
            }
        ]

        mappings = {
            "title": {"type": "text_field", "language": "es"},
            "description": {"type": "text_field", "language": "en"},
            "categories": {"type": "text_field"},  # No language specified
            "tags": {"type": "text_field"},  # No language specified
            "multimodal_content": {"type": "text_field", "language": "en"},
            "audio_description": {"type": "text_field"},  # No language specified
            "my_custom_vector": {"type": "custom_vector"},
            "embedding_vector": {"type": "custom_vector"},
            "combined_content": {
                "type": "multimodal_combination",
                "weights": {
                    "content": 0.7,
                    "image_url": 0.3
                }
            }
        }

        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=docs,
            device="cpu",
            tensor_fields=["content", "multimodal_content", "audio_description", "image_url", "combined_content",
                           "my_custom_vector", "embedding_vector"],
            mappings=mappings,
            use_existing_tensors=False,
            text_chunk_prefix="chunk:"
        )

        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            tensor_field_names=[],
            lexical_field_names=[],
            string_array_field_names=[]
        )

        # Mock external dependencies  
        mock_infer_modality.return_value = Modality.TEXT

        # Mock vespa client feed_batch response with proper structure
        mock_feed_responses = [
            FeedBatchDocumentResponse(status=200, id="doc1", message="OK"),
            FeedBatchDocumentResponse(status=200, id="doc2", message="OK"),
            FeedBatchDocumentResponse(status=200, id="doc3", message="OK")
        ]
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=mock_feed_responses,
            errors=False
        )

        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")

        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=marqo_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference,
            field_count_config=self.field_count_config
        )

        response = handler.add_documents()

        self.assertIsInstance(response, MarqoAddDocumentsResponse)
        self.assertEqual("test_index", response.index_name)
        self.assertIsInstance(response.processingTimeMs, (int, float))
        self.assertGreater(response.processingTimeMs, 0)
        self.assertGreater(len(response.items), 0)

        # Verify successful documents were processed correctly
        successful_items = [item for item in response.items if item.status == 200]
        self.assertEqual(3, len(successful_items))

        # Verify that vespa client was called for feeding documents
        self.mock_vespa_client.feed_batch.assert_called()

    def test_add_documents_with_language_on_old_index_raises_error(self):
        """Test that using language mapping on an old index raises AddDocumentsError"""
        old_marqo_index = self.semi_structured_marqo_index(
            name="old_test_index",
            marqo_version="2.15.0",  # Version before language support
            tensor_field_names=[],
            lexical_field_names=[],
            string_array_field_names=[]
        )

        docs = [
            {
                "_id": "doc1",
                "title": "Hola mundo",
                "description": "Este es un documento en español"
            }
        ]

        mappings = {
            "title": {"type": "text_field", "language": "es"}
        }

        add_docs_params = AddDocsParams(
            index_name="old_test_index",
            docs=docs,
            device="cpu",
            tensor_fields=[],
            mappings=mappings,
            use_existing_tensors=False
        )

        mock_feed_responses = [
            FeedBatchDocumentResponse(status=200, id="doc1", message="OK")
        ]
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=mock_feed_responses,
            errors=False
        )
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")

        handler = SemiStructuredAddDocumentsHandler(
            marqo_index=old_marqo_index,
            add_docs_params=add_docs_params,
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management,
            inference=self.mock_inference,
            field_count_config=self.field_count_config
        )

        response = handler.add_documents()

        self.assertIsInstance(response, MarqoAddDocumentsResponse)
        self.assertEqual("old_test_index", response.index_name)

        error_items = [item for item in response.items if item.status != 200]
        self.assertEqual(1, len(error_items), "Expected exactly one error item")

        # Check that at least one document failed with the language version error
        error_item = error_items[0]
        self.assertIn("Language is only supported for indexes created with Marqo version", str(error_item.error))
        self.assertIn("2.16.0", error_item.error)
        self.assertIn("2.15.0", error_item.error)
