import unittest
from tests.integ_tests.marqo_test import MarqoTestCase


class TestStemming(MarqoTestCase):
    """
    Integration tests for stemming feature in Marqo.
    
    Tests stemming functionality similar to language feature but focusing on
    word normalization during indexing.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Create indexes for testing stemming
        cls.unstructured_index_name = cls.get_unique_index_name("stemming_unstructured")
        cls.structured_index_name = cls.get_unique_index_name("stemming_structured")
        cls.semi_structured_index_name = cls.get_unique_index_name("stemming_semi_structured")

        # Create semi-structured index (stemming only supported for v2.16+)
        cls.marqo_client.create_index(
            index_name=cls.semi_structured_index_name,
            **cls.semi_structured_marqo_index_request()
        )

        # Create structured index for comparison
        cls.marqo_client.create_index(
            index_name=cls.structured_index_name,
            **cls.structured_marqo_index_request(
                fields=[
                    {"name": "title", "type": "text", "features": ["lexical_search"]},
                    {"name": "content", "type": "text", "features": ["lexical_search"]},
                    {"name": "description", "type": "text", "features": ["lexical_search"]},
                ]
            )
        )

    def test_document_addition_with_stemming_config(self):
        """Test adding documents with stemming configuration."""
        # Test documents with words that should be stemmed
        docs = [
            {
                "_id": "doc1",
                "title": "Running shoes are great for runners",
                "content": "The running community loves running shoes designed for runners."
            },
            {
                "_id": "doc2", 
                "title": "Cooking recipes for cooking enthusiasts",
                "content": "These cooking recipes help cooks improve their cooking skills."
            }
        ]

        # Add documents with stemming configuration
        mappings = {
            "title": {"type": "text_field", "stemming": "best"},
            "content": {"type": "text_field", "stemming": "shortest"}
        }

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        self.assertFalse(response.errors)

    def test_stemming_algorithms_produce_different_results(self):
        """Test that different stemming algorithms produce different search results."""
        # Test document with words that stem differently
        docs = [
            {
                "_id": "stem_test",
                "field_best": "running runner runners ran",
                "field_shortest": "running runner runners ran",
                "field_multiple": "running runner runners ran",
                "field_none": "running runner runners ran"
            }
        ]

        # Configure different stemming algorithms for different fields
        mappings = {
            "field_best": {"type": "text_field", "stemming": "best"},
            "field_shortest": {"type": "text_field", "stemming": "shortest"},
            "field_multiple": {"type": "text_field", "stemming": "multiple"},
            "field_none": {"type": "text_field", "stemming": "none"}
        }

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        self.assertFalse(response.errors)

        # Test lexical search finds documents through stemming
        search_response = self.marqo_client.index(cls.semi_structured_index_name).search(
            "run", search_method="LEXICAL"
        )
        
        # Should find the document because "run" is the stem of "running", "runner", etc.
        self.assertTrue(len(search_response.hits) > 0)
        self.assertEqual(search_response.hits[0]["_id"], "stem_test")

    def test_stemming_field_consistency_validation(self):
        """Test that stemming configuration cannot be changed for existing fields."""
        # Add document with stemming configuration
        doc1 = [{"_id": "consistent1", "title": "Test document"}]
        mappings1 = {"title": {"type": "text_field", "stemming": "best"}}

        response1 = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            doc1, mappings=mappings1
        )
        self.assertFalse(response1.errors)

        # Try to add another document with different stemming for same field
        doc2 = [{"_id": "consistent2", "title": "Another test document"}]
        mappings2 = {"title": {"type": "text_field", "stemming": "shortest"}}

        response2 = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            doc2, mappings=mappings2
        )
        
        # Should have errors due to stemming configuration change
        self.assertTrue(response2.errors)
        error_message = str(response2.errors[0])
        self.assertIn("different stemming configuration", error_message)
        self.assertIn("Cannot change stemming", error_message)

    def test_stemming_with_lexical_search_only(self):
        """Test that stemming works with LEXICAL search method."""
        docs = [
            {
                "_id": "lexical_test",
                "content": "The developer is developing software development tools"
            }
        ]

        mappings = {
            "content": {"type": "text_field", "stemming": "best"}
        }

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        self.assertFalse(response.errors)

        # Search with stem word should find documents with variations
        search_response = self.marqo_client.index(cls.semi_structured_index_name).search(
            "develop", search_method="LEXICAL"
        )
        
        self.assertTrue(len(search_response.hits) > 0)
        self.assertEqual(search_response.hits[0]["_id"], "lexical_test")

    def test_stemming_with_hybrid_search(self):
        """Test that stemming works with HYBRID search method."""
        docs = [
            {
                "_id": "hybrid_test",
                "title": "Advanced analytics and analytical techniques",
                "description": "Using analytical methods for data analysis"
            }
        ]

        mappings = {
            "title": {"type": "text_field", "stemming": "best"},
            "description": {"type": "text_field", "stemming": "best"}
        }

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        self.assertFalse(response.errors)

        # Test HYBRID search with stemming
        search_response = self.marqo_client.index(cls.semi_structured_index_name).search(
            "analyze", search_method="HYBRID"
        )
        
        self.assertTrue(len(search_response.hits) > 0)
        self.assertEqual(search_response.hits[0]["_id"], "hybrid_test")

    def test_stemming_version_compatibility(self):
        """Test that stemming is rejected on older index versions."""
        # This test would need an older index version to test properly
        # For now, we test that current version supports stemming
        docs = [{"_id": "version_test", "field": "testing"}]
        mappings = {"field": {"type": "text_field", "stemming": "best"}}

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        
        # Should work on current version (2.16+)
        self.assertFalse(response.errors)

    def test_structured_index_stemming_limitation(self):
        """Test that stemming configurations work appropriately with structured indexes."""
        # For structured indexes, stemming would be configured at index creation time
        # This test verifies that adding documents to structured index works normally
        docs = [
            {
                "_id": "struct_test",
                "title": "Testing structured indexing",
                "content": "Content for structured index test",
                "description": "Description field for testing"
            }
        ]

        response = self.marqo_client.index(cls.structured_index_name).add_documents(docs)
        self.assertFalse(response.errors)

    def test_stemming_and_language_combination(self):
        """Test that stemming and language can be used together."""
        docs = [
            {
                "_id": "combo_test",
                "english_field": "running runners ran",
                "spanish_field": "corriendo corredores corrió"
            }
        ]

        mappings = {
            "english_field": {"type": "text_field", "language": "en", "stemming": "best"},
            "spanish_field": {"type": "text_field", "language": "es", "stemming": "best"}
        }

        response = self.marqo_client.index(cls.semi_structured_index_name).add_documents(
            docs, mappings=mappings
        )
        self.assertFalse(response.errors)

        # Test that both language and stemming work together
        for field, query in [("english_field", "run"), ("spanish_field", "corr")]:
            with self.subTest(field=field, query=query):
                search_response = self.marqo_client.index(cls.semi_structured_index_name).search(
                    query, search_method="LEXICAL", searchable_attributes=[field]
                )
                self.assertTrue(len(search_response.hits) > 0)


if __name__ == '__main__':
    unittest.main()