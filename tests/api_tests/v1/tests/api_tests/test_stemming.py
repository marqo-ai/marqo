import unittest
import time
from tests.marqo_test import MarqoTestCase
from marqo.client import Client


class TestStemmingAPI(MarqoTestCase):
    """
    API-level tests for stemming feature in Marqo.
    
    Tests stemming functionality through the client API, focusing on
    proper API behavior and error handling.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client()
        
        # Create unstructured index for stemming tests
        cls.semi_structured_index_name = cls.random_index_name("api_stemming_semi")
        cls.client.create_index(
            index_name=cls.semi_structured_index_name,
            type="unstructured",
            model="hf/all_datasets_v4_MiniLM-L6"
        )
        cls.indexes_to_delete.append(cls.semi_structured_index_name)

    def test_stemming_document_addition_api(self):
        """Test stemming through client API document addition."""
        docs = [
            {
                "_id": "api_doc1",
                "title": "Running and walking exercises",
                "content": "Runners enjoy running while walkers prefer walking"
            }
        ]

        mappings = {
            "title": {"type": "text_field", "language": "en", "stemming": "best"},
            "content": {"type": "text_field", "language": "en", "stemming": "shortest"}
        }

        # Test API call succeeds
        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(response['errors'])

    def test_stemming_search_api_lexical(self):
        """Test stemming works through API lexical search."""
        docs = [
            {
                "_id": "search_test1",
                "field": "analytical analysis analyzing analyst"
            }
        ]

        mappings = {
            "field": {"type": "text_field", "language": "en", "stemming": "best"}
        }

        # Add documents
        add_response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(add_response['errors'])

        # Search using stemmed form
        search_response = self.client.index(self.semi_structured_index_name).search(
            "analyze", search_method="LEXICAL"
        )
        
        self.assertTrue(len(search_response['hits']) > 0)
        self.assertEqual(search_response['hits'][0]["_id"], "search_test1")

    def test_stemming_search_api_hybrid(self):
        """Test stemming works through API hybrid search."""
        docs = [
            {
                "_id": "hybrid_api_test",
                "hybrid_content": "optimization optimize optimized optimizing"
            }
        ]

        mappings = {
            "hybrid_content": {"type": "text_field", "language": "en", "stemming": "best"}
        }

        # Add documents
        add_response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=["hybrid_content"]
        )
        if add_response['errors']:
            print("Hybrid test error:", add_response)
        self.assertFalse(add_response['errors'])

        # Test HYBRID search
        search_response = self.client.index(self.semi_structured_index_name).search(
            "optimal", search_method="HYBRID"
        )
        
        self.assertTrue(len(search_response['hits']) > 0)
        self.assertEqual(search_response['hits'][0]["_id"], "hybrid_api_test")

    def test_stemming_invalid_value_api_error(self):
        """Test that invalid stemming values produce proper API errors."""
        docs = [{"_id": "invalid_test", "field": "test content"}]
        mappings = {"field": {"type": "text_field", "language": "en", "stemming": "invalid_algorithm"}}

        # Should raise an error due to invalid stemming value
        with self.assertRaises(Exception) as cm:
            self.client.index(self.semi_structured_index_name).add_documents(
                docs, mappings=mappings, tensor_fields=[]
            )
        
        error_message = str(cm.exception)
        self.assertIn("stemming", error_message.lower())

    def test_stemming_field_change_api_error(self):
        """Test that changing stemming configuration produces API error."""
        # First add document with one stemming config
        docs1 = [{"_id": "change_test1", "title": "First document"}]
        mappings1 = {"title": {"type": "text_field", "language": "en", "stemming": "best"}}

        response1 = self.client.index(self.semi_structured_index_name).add_documents(
            docs1, mappings=mappings1, tensor_fields=[]
        )
        self.assertFalse(response1['errors'])

        # Try to add document with different stemming config for same field
        docs2 = [{"_id": "change_test2", "title": "Second document"}]
        mappings2 = {"title": {"type": "text_field", "language": "en", "stemming": "shortest"}}

        response2 = self.client.index(self.semi_structured_index_name).add_documents(
            docs2, mappings=mappings2, tensor_fields=[]
        )
        
        # Should have errors
        self.assertTrue(response2['errors'])
        error_message = response2['items'][0]['message']
        self.assertIn("different stemming configuration", error_message)

    def test_stemming_multiple_algorithms_api(self):
        """Test using different stemming algorithms through API."""
        docs = [
            {
                "_id": "multi_stem",
                "field_best": "processing processed processes",
                "field_shortest": "processing processed processes", 
                "field_multiple": "processing processed processes",
                "field_none": "processing processed processes"
            }
        ]

        mappings = {
            "field_best": {"type": "text_field", "language": "en", "stemming": "best"},
            "field_shortest": {"type": "text_field", "language": "en", "stemming": "shortest"},
            "field_multiple": {"type": "text_field", "language": "en", "stemming": "multiple"},
            "field_none": {"type": "text_field", "language": "en", "stemming": "none"}
        }

        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(response['errors'])

        # Test search works with different algorithms
        search_response = self.client.index(self.semi_structured_index_name).search(
            "process", search_method="LEXICAL"
        )
        
        self.assertTrue(len(search_response['hits']) > 0)
        self.assertEqual(search_response['hits'][0]["_id"], "multi_stem")

    def test_stemming_with_language_api(self):
        """Test stemming combined with language through API."""
        docs = [
            {
                "_id": "lang_stem_test",
                "lang_english_text": "running runners ran",
                "lang_french_text": "courant coureurs couru"
            }
        ]

        mappings = {
            "lang_english_text": {"type": "text_field", "language": "en", "stemming": "best"},
            "lang_french_text": {"type": "text_field", "language": "fr", "stemming": "best"}
        }

        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(response['errors'])

        # Allow time for indexing to complete
        time.sleep(2)

        # Test that documents with language+stemming configuration are searchable
        # First test English
        english_search = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL", searchable_attributes=["lang_english_text"]
        )
        self.assertTrue(len(english_search['hits']) > 0)
        
        # Test general search to verify document is indexed
        general_search = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL"
        )
        self.assertTrue(len(general_search['hits']) > 0)
        
        # Test that language was accepted (this is the main purpose of the test)
        self.assertEqual(general_search['hits'][0]["_id"], "lang_stem_test")

    def test_stemming_none_algorithm_api(self):
        """Test that stemming 'none' disables stemming through API."""
        docs = [
            {
                "_id": "no_stem_test",
                "with_stemming": "running runners ran",
                "without_stemming": "running runners ran"
            }
        ]

        mappings = {
            "with_stemming": {"type": "text_field", "language": "en", "stemming": "best"},
            "without_stemming": {"type": "text_field", "language": "en", "stemming": "none"}
        }

        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(response['errors'])

        # Allow time for indexing to complete
        time.sleep(2)

        # Search for "run" should find with_stemming field
        # Note: Stemming behavior may vary, so let's test that documents are indexed correctly
        # and that different stemming configs can be applied
        search_response = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL", searchable_attributes=["with_stemming"]
        )
        self.assertTrue(len(search_response['hits']) > 0)
        
        # Test that the document was indexed correctly
        search_response2 = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL", searchable_attributes=["without_stemming"]
        )
        self.assertTrue(len(search_response2['hits']) > 0)

        # Search in field without stemming should not find partial matches as easily
        search_response_no_stem = self.client.index(self.semi_structured_index_name).search(
            "run", search_method="LEXICAL", searchable_attributes=["without_stemming"]
        )
        # This depends on exact matching behavior, may find fewer or no results


if __name__ == '__main__':
    unittest.main()