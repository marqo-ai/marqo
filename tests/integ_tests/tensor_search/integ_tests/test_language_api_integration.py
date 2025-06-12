"""
Integration tests for language functionality in semi-structured indexes.
Tests the complete workflow from API to Vespa integration.
"""
import uuid
from unittest import mock
import os

import pytest

from marqo.api.exceptions import InvalidArgError
from marqo.core.models.marqo_index import *
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLanguageAPIIntegration(MarqoTestCase):
    """Integration tests for language functionality in semi-structured indexes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        
        # Create a semi-structured index for language testing
        cls.semi_structured_index_request = cls.unstructured_marqo_index_request(
            name='test_language_' + str(uuid.uuid4()).replace('-', '')
        )
        
        # Create an unstructured index for validation testing  
        cls.unstructured_index_request = cls.unstructured_marqo_index_request(
            name='test_unstructured_' + str(uuid.uuid4()).replace('-', '')
        )
        
        cls.indexes = cls.create_indexes([
            cls.semi_structured_index_request,
            cls.unstructured_index_request
        ])
        
        cls.semi_structured_index_name = cls.semi_structured_index_request.name
        cls.unstructured_index_name = cls.unstructured_index_request.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)
        
        # Set device to CPU for all tests
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_add_documents_with_language_mapping(self):
        """Test adding documents with language mapping specification."""
        docs = [
            {
                "_id": "doc1",
                "title": "Hola mundo",
                "description": "Este es un documento en español"
            },
            {
                "_id": "doc2", 
                "title": "Hello world",
                "description": "This is an English document"
            }
        ]
        
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "es"
            },
            "description": {
                "type": "text_field_language", 
                "language": "en"
            }
        }
        
        # Add documents with language mappings
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=["title"],
                mappings=mappings
            )
        )
        
        # Verify successful addition
        self.assertFalse(response.errors)
        self.assertEqual(len(response.items), 2)
        
        # Verify documents can be retrieved
        retrieved_doc = tensor_search.get_document_by_id(
            config=self.config,
            index_name=self.semi_structured_index_name,
            document_id="doc1"
        )
        
        self.assertEqual(retrieved_doc["title"], "Hola mundo")
        self.assertEqual(retrieved_doc["description"], "Este es un documento en español")

    def test_search_with_language_override_lexical(self):
        """Test lexical search with language override."""
        # Add Spanish documents first
        docs = [
            {
                "_id": "es1",
                "content": "Los gatos son animales domésticos"
            },
            {
                "_id": "es2", 
                "content": "El perro corre rápidamente"
            }
        ]
        
        mappings = {
            "content": {
                "type": "text_field_language",
                "language": "es"
            }
        }
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu", 
                tensor_fields=[],
                mappings=mappings
            )
        )
        
        # Test lexical search with language override
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_index_name,
            text="gatos",
            search_method=SearchMethod.LEXICAL,
            model={"language": "es"}
        )
        
        self.assertGreater(len(search_result["hits"]), 0)
        self.assertEqual(search_result["hits"][0]["_id"], "es1")

    def test_search_with_language_override_hybrid(self):
        """Test hybrid search with language override."""
        # Add documents with different languages
        docs = [
            {
                "_id": "fr1",
                "title": "Bonjour le monde", 
                "content": "Ceci est un document français"
            },
            {
                "_id": "fr2",
                "title": "Chat noir",
                "content": "Un chat noir très mignon"
            }
        ]
        
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "fr"
            },
            "content": {
                "type": "text_field_language",
                "language": "fr"
            }
        }
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=["title"],
                mappings=mappings
            )
        )
        
        # Test hybrid search with language override
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_index_name,
            text="chat",
            search_method=SearchMethod.HYBRID,
            model={"language": "fr"}
        )
        
        self.assertGreater(len(search_result["hits"]), 0)

    def test_language_validation_invalid_codes(self):
        """Test validation of invalid language codes."""
        docs = [{"_id": "test1", "title": "Test content"}]
        
        # Test invalid language codes
        invalid_mappings = [
            {
                "title": {
                    "type": "text_field_language",
                    "language": "invalid_lang_code"
                }
            },
            {
                "title": {
                    "type": "text_field_language", 
                    "language": "1234"
                }
            },
            {
                "title": {
                    "type": "text_field_language",
                    "language": ""
                }
            }
        ]
        
        for invalid_mapping in invalid_mappings:
            with self.subTest(mapping=invalid_mapping):
                response = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.semi_structured_index_name,
                        docs=docs,
                        device="cpu",
                        tensor_fields=[],
                        mappings=invalid_mapping
                    )
                )
                
                # Should have errors for invalid language codes
                self.assertTrue(response.errors)

    def test_language_with_tensor_search_validation(self):
        """Test that language override is rejected for tensor search."""
        # Add some documents first
        docs = [{"_id": "test1", "title": "Test content"}]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=["title"]
            )
        )
        
        # Test that tensor search with language parameter raises error
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.TENSOR,
            model={"language": "en"}
        )
        
        with self.assertRaisesRegex(Exception, "model.language.*not.*supported.*tensor"):
            # This should be caught by validation in the search endpoint
            tensor_search.search(
                config=self.config,
                index_name=self.semi_structured_index_name,
                text=search_query.q,
                search_method=search_query.searchMethod,
                model=search_query.model
            )

    def test_language_with_unstructured_index_validation(self):
        """Test that language mapping is rejected for unstructured indexes."""
        docs = [{"_id": "test1", "title": "Test content"}]
        
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "en"
            }
        }
        
        # Should fail because unstructured indexes don't support language mappings
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=[],
                mappings=mappings
            )
        )
        
        # Should have errors for using language mapping with unstructured index
        self.assertTrue(response.errors)

    def test_language_mapping_validation_non_text_fields(self):
        """Test that language mappings are rejected for non-text fields."""
        docs = [
            {
                "_id": "test1", 
                "numeric_field": 123,
                "boolean_field": True,
                "list_field": ["item1", "item2"]
            }
        ]
        
        invalid_mappings = [
            {
                "numeric_field": {
                    "type": "text_field_language",
                    "language": "en"
                }
            },
            {
                "boolean_field": {
                    "type": "text_field_language",
                    "language": "en"
                }
            },
            {
                "list_field": {
                    "type": "text_field_language",
                    "language": "en"
                }
            }
        ]
        
        for invalid_mapping in invalid_mappings:
            with self.subTest(mapping=invalid_mapping):
                response = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.semi_structured_index_name,
                        docs=docs,
                        device="cpu",
                        tensor_fields=[],
                        mappings=invalid_mapping
                    )
                )
                
                # Should have errors for applying language mapping to non-text fields
                self.assertTrue(response.errors)

    def test_multiple_documents_different_languages(self):
        """Test adding multiple documents with different language specifications."""
        docs = [
            {
                "_id": "multi1",
                "title_en": "English title",
                "title_es": "Título en español",
                "title_fr": "Titre français"
            },
            {
                "_id": "multi2", 
                "title_en": "Another English title",
                "title_es": "Otro título español",
                "title_fr": "Autre titre français"
            }
        ]
        
        mappings = {
            "title_en": {
                "type": "text_field_language",
                "language": "en"
            },
            "title_es": {
                "type": "text_field_language",
                "language": "es"
            },
            "title_fr": {
                "type": "text_field_language",
                "language": "fr"
            }
        }
        
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=[],
                mappings=mappings
            )
        )
        
        # Should successfully add documents with multiple language specifications
        self.assertFalse(response.errors)
        self.assertEqual(len(response.items), 2)
        
        # Verify document retrieval
        retrieved_doc = tensor_search.get_document_by_id(
            config=self.config,
            index_name=self.semi_structured_index_name,
            document_id="multi1"
        )
        
        self.assertEqual(retrieved_doc["title_en"], "English title")
        self.assertEqual(retrieved_doc["title_es"], "Título en español")
        self.assertEqual(retrieved_doc["title_fr"], "Titre français")

    def test_search_language_override_with_different_methods(self):
        """Test language override works correctly with different search methods."""
        # Add multilingual content
        docs = [
            {
                "_id": "search1",
                "content": "running quickly through the park"
            },
            {
                "_id": "search2",
                "content": "corriendo rápidamente por el parque"
            }
        ]
        
        mappings = {
            "content": {
                "type": "text_field_language",
                "language": "en"
            }
        }
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs,
                device="cpu",
                tensor_fields=["content"],
                mappings=mappings
            )
        )
        
        # Test lexical search with English language
        lexical_result = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_index_name,
            text="running",
            search_method=SearchMethod.LEXICAL,
            model={"language": "en"}
        )
        
        self.assertGreater(len(lexical_result["hits"]), 0)
        
        # Test hybrid search with Spanish language  
        hybrid_result = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_index_name,
            text="corriendo",
            search_method=SearchMethod.HYBRID,
            model={"language": "es"}
        )
        
        self.assertGreater(len(hybrid_result["hits"]), 0)

    def test_field_evolution_with_language_mappings(self):
        """Test that field evolution works correctly with language mappings."""
        # First, add a document without language mapping
        docs1 = [
            {
                "_id": "evolve1",
                "title": "Initial title without language"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs1,
                device="cpu",
                tensor_fields=[]
            )
        )
        
        # Then add a document with language mapping for the same field
        docs2 = [
            {
                "_id": "evolve2", 
                "title": "Second title with language"
            }
        ]
        
        mappings = {
            "title": {
                "type": "text_field_language",
                "language": "en"
            }
        }
        
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_name,
                docs=docs2,
                device="cpu", 
                tensor_fields=[],
                mappings=mappings
            )
        )
        
        # Should successfully handle field evolution
        self.assertFalse(response.errors)
        
        # Verify both documents exist
        doc1 = tensor_search.get_document_by_id(
            config=self.config,
            index_name=self.semi_structured_index_name,
            document_id="evolve1"
        )
        
        doc2 = tensor_search.get_document_by_id(
            config=self.config,
            index_name=self.semi_structured_index_name,
            document_id="evolve2"
        )
        
        self.assertEqual(doc1["title"], "Initial title without language")
        self.assertEqual(doc2["title"], "Second title with language")