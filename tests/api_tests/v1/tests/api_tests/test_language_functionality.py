"""
API tests for language functionality in semi-structured indexes.
Tests the language feature through the actual HTTP API endpoints.
"""
import uuid

from marqo.errors import MarqoWebError
from tests.marqo_test import MarqoTestCase


class TestLanguageFunctionality(MarqoTestCase):
    """API tests for language functionality in semi-structured indexes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        
        # Create a semi-structured index for language testing
        cls.semi_structured_index_name = "test_language_" + str(uuid.uuid4()).replace('-', '')
        
        # Create an unstructured index for validation testing  
        cls.unstructured_index_name = "test_unstructured_" + str(uuid.uuid4()).replace('-', '')
        
        # Create a structured index for validation testing
        cls.structured_index_name = "test_structured_" + str(uuid.uuid4()).replace('-', '')
        
        cls.create_indexes([
            {
                "indexName": cls.semi_structured_index_name,
                "type": "unstructured",  # Semi-structured is created as unstructured
                "model": "hf/all-MiniLM-L6-v2"
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2"
            },
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            }
        ])
        
        cls.indexes_to_delete = [
            cls.semi_structured_index_name,
            cls.unstructured_index_name, 
            cls.structured_index_name
        ]

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
        
        # Add documents with language mappings
        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=["title"],
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "es"
                },
                "description": {
                    "type": "text_field_language", 
                    "language": "en"
                }
            }
        )
        
        # Verify successful addition
        self.assertFalse(response["errors"])
        self.assertEqual(len(response["items"]), 2)
        
        # Verify documents can be retrieved
        retrieved_doc = self.client.index(self.semi_structured_index_name).get_document("doc1")
        
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
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=[],
            mappings={
                "content": {
                    "type": "text_field_language",
                    "language": "es"
                }
            }
        )
        
        # Test lexical search with language override
        search_result = self.client.index(self.semi_structured_index_name).search(
            q="gatos",
            search_method="LEXICAL",
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
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=["title"],
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "fr"
                },
                "content": {
                    "type": "text_field_language",
                    "language": "fr"
                }
            }
        )
        
        # Test hybrid search with language override
        search_result = self.client.index(self.semi_structured_index_name).search(
            q="chat",
            search_method="HYBRID",
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
                response = self.client.index(self.semi_structured_index_name).add_documents(
                    docs,
                    tensor_fields=[],
                    mappings=invalid_mapping
                )
                
                # Should have errors for invalid language codes
                self.assertTrue(response["errors"])

    def test_language_with_tensor_search_validation(self):
        """Test that language override is rejected for tensor search."""
        # Add some documents first
        docs = [{"_id": "test1", "title": "Test content"}]
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=["title"]
        )
        
        # Test that tensor search with language parameter raises error
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.semi_structured_index_name).search(
                q="test query",
                search_method="TENSOR",
                model={"language": "en"}
            )
        
        error_msg = str(cm.exception)
        self.assertIn("model.language", error_msg)
        self.assertIn("not", error_msg.lower())
        self.assertIn("tensor", error_msg.lower())

    def test_language_with_unstructured_index_validation(self):
        """Test that language mapping is rejected for unstructured indexes."""
        docs = [{"_id": "test1", "title": "Test content"}]
        
        # Should fail because unstructured indexes don't support language mappings
        response = self.client.index(self.unstructured_index_name).add_documents(
            docs,
            tensor_fields=[],
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "en"
                }
            }
        )
        
        # Should have errors for using language mapping with unstructured index
        self.assertTrue(response["errors"])

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
                response = self.client.index(self.semi_structured_index_name).add_documents(
                    docs,
                    tensor_fields=[],
                    mappings=invalid_mapping
                )
                
                # Should have errors for applying language mapping to non-text fields
                self.assertTrue(response["errors"])

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
        
        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=[],
            mappings={
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
        )
        
        # Should successfully add documents with multiple language specifications
        self.assertFalse(response["errors"])
        self.assertEqual(len(response["items"]), 2)
        
        # Verify document retrieval
        retrieved_doc = self.client.index(self.semi_structured_index_name).get_document("multi1")
        
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
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=["content"],
            mappings={
                "content": {
                    "type": "text_field_language",
                    "language": "en"
                }
            }
        )
        
        # Test lexical search with English language
        lexical_result = self.client.index(self.semi_structured_index_name).search(
            q="running",
            search_method="LEXICAL",
            model={"language": "en"}
        )
        
        self.assertGreater(len(lexical_result["hits"]), 0)
        
        # Test hybrid search with Spanish language  
        hybrid_result = self.client.index(self.semi_structured_index_name).search(
            q="corriendo",
            search_method="HYBRID",
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
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs1,
            tensor_fields=[]
        )
        
        # Then add a document with language mapping for the same field
        docs2 = [
            {
                "_id": "evolve2", 
                "title": "Second title with language"
            }
        ]
        
        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs2,
            tensor_fields=[],
            mappings={
                "title": {
                    "type": "text_field_language",
                    "language": "en"
                }
            }
        )
        
        # Should successfully handle field evolution
        self.assertFalse(response["errors"])
        
        # Verify both documents exist
        doc1 = self.client.index(self.semi_structured_index_name).get_document("evolve1")
        doc2 = self.client.index(self.semi_structured_index_name).get_document("evolve2")
        
        self.assertEqual(doc1["title"], "Initial title without language")
        self.assertEqual(doc2["title"], "Second title with language")

    def test_language_search_validation_for_structured_index(self):
        """Test that language search parameter is rejected for structured indexes."""
        # Add some documents to structured index
        docs = [{"_id": "test1", "title": "Test content", "content": "Some content"}]
        
        self.client.index(self.structured_index_name).add_documents(docs)
        
        # Test that structured index search with language parameter raises error
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index(self.structured_index_name).search(
                q="test query",
                search_method="LEXICAL",
                model={"language": "en"}
            )
        
        error_msg = str(cm.exception)
        self.assertIn("model.language", error_msg)
        self.assertIn("semi-structured", error_msg.lower())

    def test_valid_language_codes(self):
        """Test that valid language codes are accepted."""
        docs = [{"_id": "test1", "title": "Test content"}]
        
        # Test various valid language codes
        valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
        
        for lang in valid_languages:
            with self.subTest(language=lang):
                response = self.client.index(self.semi_structured_index_name).add_documents(
                    [{"_id": f"test_{lang}", "title": f"Content in {lang}"}],
                    tensor_fields=[],
                    mappings={
                        "title": {
                            "type": "text_field_language",
                            "language": lang
                        }
                    }
                )
                
                # Should succeed with valid language codes
                self.assertFalse(response["errors"])

    def test_language_search_with_no_language_mapping(self):
        """Test that language search works even when documents have no language mapping."""
        # Add documents without language mapping
        docs = [
            {
                "_id": "no_lang1",
                "content": "This is content without language mapping"
            }
        ]
        
        self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=[]
        )
        
        # Search should still work with language override
        search_result = self.client.index(self.semi_structured_index_name).search(
            q="content",
            search_method="LEXICAL",
            model={"language": "en"}
        )
        
        self.assertGreater(len(search_result["hits"]), 0)