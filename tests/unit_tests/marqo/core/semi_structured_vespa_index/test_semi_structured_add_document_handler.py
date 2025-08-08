from unittest.mock import Mock, patch

import numpy as np

from marqo.core.exceptions import InvalidArgumentError
from marqo.core.models.marqo_index import Stemming
from marqo.core.inference.api import InferenceRequest, InferenceResult, Modality
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse
from marqo.core.models.marqo_index import CollapseField
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import (
    SemiStructuredAddDocumentsHandler,
    SemiStructuredFieldCountConfig
)
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

        # Setup vespa client mock for translate_vespa_document_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")

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
            "title": {"type": "text_field", "language": "es", "stemming": "best"},
            "description": {"type": "text_field", "language": "en", "stemming": "shortest"},
            "categories": {"type": "text_field", "language": "en", "stemming": "multiple"},
            "tags": {"type": "text_field", "language": "en", "stemming": "none"},
            "multimodal_content": {"type": "text_field", "language": "en"},
            "audio_description": {"type": "text_field", "language": "en"},
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

        mappings_cases = [mappings, None]
        for mappings in mappings_cases:
            with self.subTest(mappings=mappings):
                add_docs_params = AddDocsParams(
                    index_name="test_index",
                    docs=docs,
                    device="cpu",
                    tensor_fields=["content", "multimodal_content", "audio_description", "image_url",
                                   "combined_content",
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

    def test_add_documents_with_language_or_stemming_on_old_index_raises_error(self):
        """Test that using language and/or stemming mapping on an old index raises AddDocumentsError"""
        old_marqo_index = self.semi_structured_marqo_index(
            name="old_test_index",
            marqo_version="2.15.0",  # Version before language/stemming support
            tensor_field_names=[],
            lexical_field_names=[],
            string_array_field_names=[]
        )

        docs = [{"_id": "doc1", "title": "Test document"}]

        # Mock feed response
        mock_feed_responses = [FeedBatchDocumentResponse(status=200, id="doc1", message="OK")]
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=mock_feed_responses, errors=False
        )
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")

        test_cases = [
            ("language_only", {"type": "text_field", "language": "es"}, "Language is only supported"),
            ("stemming_only", {"type": "text_field", "stemming": "best"}, "Stemming is only supported"),
            ("both", {"type": "text_field", "language": "es", "stemming": "best"}, "Language is only supported")
        ]

        for case_name, mapping, expected_error_text in test_cases:
            with self.subTest(case=case_name):
                add_docs_params = AddDocsParams(
                    index_name="old_test_index",
                    docs=docs,
                    device="cpu",
                    tensor_fields=[],
                    mappings={"title": mapping},
                    use_existing_tensors=False
                )

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
                self.assertEqual(1, len(error_items), f"Expected exactly one error item for {case_name}")

                error_item = error_items[0]
                self.assertIn(expected_error_text, str(error_item.error))
                self.assertIn("2.16.0", error_item.error)
                self.assertIn("2.15.0", error_item.error)

    def test_text_field_mapping_without_language_or_stemming_raises_invalid_argument_error(self):
        """Test that text_field mapping without language or stemming specification raises InvalidArgumentError"""
        docs = [{"_id": "doc1", "title": "Test document"}]
        mappings = {"title": {"type": "text_field"}}  # Missing language specification

        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=docs,
            device="cpu",
            tensor_fields=[],
            mappings=mappings,
            use_existing_tensors=False
        )

        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            tensor_field_names=[],
            lexical_field_names=[],
            string_array_field_names=[]
        )

        with self.assertRaises(InvalidArgumentError) as cm:
            SemiStructuredAddDocumentsHandler(
                marqo_index=marqo_index,
                add_docs_params=add_docs_params,
                vespa_client=self.mock_vespa_client,
                index_management=self.mock_index_management,
                inference=self.mock_inference,
                field_count_config=self.field_count_config
            )

        error_message = str(cm.exception)
        self.assertIn("text_field", error_message)
        self.assertIn("language", error_message.lower())
        self.assertIn("stemming", error_message.lower())

    def test_language_and_stemming_field_consistency_validation(self):
        """Test that stemming and language configuration cannot be changed for existing fields"""
        docs = [{"_id": "doc1", "title": "Test document"}]

        marqo_index = self.semi_structured_marqo_index(
            name="consistency_test_index",
            tensor_field_names=[],
            lexical_field_names=["title"],  # Field already exists
            string_array_field_names=[]
        )

        # Mock feed response
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=[], errors=False
        )

        from marqo.core.models.marqo_index import Field

        test_cases = [
            (
                "stemming_change",
                {"stemming": Stemming.Best, "language": "en"},  # Existing field config
                {"type": "text_field", "stemming": "shortest", "language": "en"},  # New mapping
                ["different stemming configuration", "Cannot change stemming"]
            ),
            (
                "language_change",
                {"stemming": Stemming.Best, "language": "en"},  # Existing field config
                {"type": "text_field", "stemming": "best", "language": "es"},  # New mapping
                ["different language configuration", "Cannot change language"]
            ),
            (
                "both_change",
                {"stemming": Stemming.Best, "language": "en"},  # Existing field config
                {"type": "text_field", "stemming": "shortest", "language": "es"},  # New mapping
                ["different language configuration", "Cannot change language"]  # Language error comes first
            )
        ]

        for case_name, existing_config, new_mapping, expected_errors in test_cases:
            with self.subTest(case=case_name):
                # Set up existing field with specific configuration
                existing_field = marqo_index.field_map["title"]
                field_with_config = Field(
                    name=existing_field.name,
                    type=existing_field.type,
                    features=existing_field.features,
                    lexical_field_name=existing_field.lexical_field_name,
                    filter_field_name=existing_field.filter_field_name,
                    dependent_fields=existing_field.dependent_fields,
                    language=existing_config.get("language"),
                    stemming=existing_config.get("stemming")
                )
                marqo_index.field_map["title"] = field_with_config

                add_docs_params = AddDocsParams(
                    index_name="consistency_test_index",
                    docs=docs,
                    device="cpu",
                    tensor_fields=[],
                    mappings={"title": new_mapping},
                    use_existing_tensors=False
                )

                handler = SemiStructuredAddDocumentsHandler(
                    marqo_index=marqo_index,
                    add_docs_params=add_docs_params,
                    vespa_client=self.mock_vespa_client,
                    index_management=self.mock_index_management,
                    inference=self.mock_inference,
                    field_count_config=self.field_count_config
                )

                response = handler.add_documents()

                # Should have errors due to configuration mismatch
                error_items = [item for item in response.items if item.status != 200]
                self.assertEqual(1, len(error_items), f"Expected exactly one error item for {case_name}")

                error_item = error_items[0]
                error_message = str(error_item.error)
                for expected_error in expected_errors:
                    self.assertIn(expected_error, error_message,
                                  f"Expected '{expected_error}' in error message for {case_name}")

    def test_stemming_value_validation(self):
        """Test that invalid stemming values raise appropriate errors"""
        docs = [{"_id": "doc1", "title": "Test document"}]

        marqo_index = self.semi_structured_marqo_index(
            name="stemming_validation_test",
            tensor_field_names=[],
            lexical_field_names=[],
            string_array_field_names=[]
        )

        add_docs_params = AddDocsParams(
            index_name="stemming_validation_test",
            docs=docs,
            device="cpu",
            tensor_fields=[],
            mappings={
                "title": {"type": "text_field", "stemming": "invalid_algorithm"}
            },
            use_existing_tensors=False
        )

        # Mock feed response
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=[], errors=False
        )

        # Should raise InvalidArgumentError during initialization due to invalid stemming value
        with self.assertRaises(InvalidArgumentError) as cm:
            handler = SemiStructuredAddDocumentsHandler(
                marqo_index=marqo_index,
                add_docs_params=add_docs_params,
                vespa_client=self.mock_vespa_client,
                index_management=self.mock_index_management,
                inference=self.mock_inference,
                field_count_config=self.field_count_config
            )

        error_message = str(cm.exception)
        self.assertIn("is not one of", error_message)
        self.assertIn("invalid_algorithm", error_message)

    def test_collapse_field_validation_should_succeed(self):
        """Test collapse field validation with various success scenarios"""
        collapse_fields = [CollapseField(name="parent_id", minGroups=100)]

        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=[{"_id": "doc1", "title": "Test document", "parent_id": "product_123"}],
            tensor_fields=[],
        )

        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            collapse_fields=collapse_fields
        )

        # Mock the vespa client response for successful documents
        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=[FeedBatchDocumentResponse(status=200, id="doc1", message="OK")],
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
        error_items = [item for item in response.items if item.status != 200]
        self.assertEqual(0, len(error_items))

    def test_collapse_field_validation_should_fail(self):
        """Test collapse field validation with various failure scenarios"""
        collapse_fields = [CollapseField(name="parent_id", minGroups=100)]
        
        test_cases = [
            {
                "name": "missing_field",
                "doc": {"_id": "doc1", "title": "Test document"},
                "expected_error": "Document missing required field 'parent_id'",
            },
            {
                "name": "invalid_type_int",
                "doc": {"_id": "doc2", "title": "Test document", "parent_id": 123},
                "expected_error": "Field 'parent_id' must be of type string",
            },
            {
                "name": "invalid_type_none",
                "doc": {"_id": "doc3", "title": "Test document", "parent_id": None},
                "expected_error": "Field 'parent_id' must be of type string",
            },
            {
                "name": "empty_string",
                "doc": {"_id": "doc4", "title": "Test document", "parent_id": ""},
                "expected_error": "Field 'parent_id' cannot be empty",
            },
            {
                "name": "whitespace_only",
                "doc": {"_id": "doc5", "title": "Test document", "parent_id": "   "},
                "expected_error": "Field 'parent_id' cannot be empty",
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                add_docs_params = AddDocsParams(
                    index_name="test_index",
                    docs=[case["doc"]],
                    tensor_fields=[],
                )

                marqo_index = self.semi_structured_marqo_index(
                    name="test_index",
                    collapse_fields=collapse_fields
                )

                handler = SemiStructuredAddDocumentsHandler(
                    marqo_index=marqo_index,
                    add_docs_params=add_docs_params,
                    vespa_client=self.mock_vespa_client,
                    index_management=self.mock_index_management,
                    inference=self.mock_inference,
                    field_count_config=self.field_count_config
                )

                response = handler.add_documents()
                error_items = [item for item in response.items if item.status != 200]
                self.assertEqual(1, len(error_items), f"Expected error for case: {case['name']}")
                self.assertIn(case["expected_error"], str(error_items[0].error))

    def test_collapse_field_validation_no_collapse_fields_configured(self):
        """Test that validation is skipped when no collapse fields are configured"""
        docs = [{"_id": "doc1", "title": "Test document"}]
        
        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=docs,
            tensor_fields=[],
        )

        marqo_index = self.semi_structured_marqo_index(
            name="test_index",
            collapse_fields=None
        )

        self.mock_vespa_client.feed_batch.return_value = FeedBatchResponse(
            responses=[FeedBatchDocumentResponse(status=200, id="doc1", message="OK")],
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
        error_items = [item for item in response.items if item.status != 200]
        self.assertEqual(0, len(error_items))
