"""
API tests for language functionality in semi-structured indexes.
Tests the language feature through the actual HTTP API endpoints using requests.
"""
import uuid
import requests
import unittest


class TestLanguageFunctionality(unittest.TestCase):
    """API tests for language functionality in semi-structured indexes using requests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.marqo_url = "http://localhost:8882"
        cls.indexes_to_delete = []

    def setUp(self) -> None:
        # Create a unique index for each test to avoid field conflicts
        self.test_index_name = "test_language_" + str(uuid.uuid4()).replace('-', '')
        
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}",
            json={
                "type": "unstructured",  # Semi-structured is created as unstructured
                "model": "hf/all-MiniLM-L6-v2"
            }
        )
        response.raise_for_status()
        self.__class__.indexes_to_delete.append(self.test_index_name)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up indexes
        for index_name in cls.indexes_to_delete:
            try:
                requests.delete(f"{cls.marqo_url}/indexes/{index_name}")
            except:
                pass

    def test_french_language_search_integration(self):
        """Test that language works correctly by indexing French text and searching for it."""
        # Add French document with language mapping
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/documents",
            json={
                "documents": [
                    {
                        "_id": "french_doc",
                        "title": "Innovation en Intelligence Artificielle"
                    }
                ],
                "tensorFields": [],
                "mappings": {
                    "title": {
                        "type": "text_field",
                        "language": "fr"
                    }
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        self.assertFalse(result["errors"])
        
        # Search for "Intelligence" with French language
        search_response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/search",
            json={
                "q": "Intelligence",
                "searchMethod": "LEXICAL",
                "language": "fr"
            }
        )
        search_response.raise_for_status()
        search_result = search_response.json()
        
        # Debug: print search result to see what's happening
        print(f"Search result: {search_result}")
        
        # Verify the French document is returned
        self.assertGreater(len(search_result["hits"]), 0, f"Expected hits but got: {search_result}")
        self.assertEqual(search_result["hits"][0]["_id"], "french_doc")
        self.assertEqual(search_result["hits"][0]["title"], "Innovation en Intelligence Artificielle")

    def test_add_documents_with_language_mapping(self):
        """Test adding documents with language mapping specification."""
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/documents",
            json={
                "documents": [
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
                ],
                "tensorFields": ["title"],
                "mappings": {
                    "title": {
                        "type": "text_field",
                        "language": "es"
                    },
                    "description": {
                        "type": "text_field", 
                        "language": "en"
                    }
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Verify successful addition
        self.assertFalse(result["errors"])
        self.assertEqual(len(result["items"]), 2)
        
        # Verify documents can be retrieved
        get_response = requests.get(f"{self.marqo_url}/indexes/{self.test_index_name}/documents/doc1")
        get_response.raise_for_status()
        retrieved_doc = get_response.json()
        
        self.assertEqual(retrieved_doc["title"], "Hola mundo")
        self.assertEqual(retrieved_doc["description"], "Este es un documento en español")

    def test_search_with_language_override_lexical(self):
        """Test lexical search with language override."""
        # Add Spanish documents first
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/documents",
            json={
                "documents": [
                    {
                        "_id": "es1",
                        "content": "Los gatos son animales domésticos"
                    },
                    {
                        "_id": "es2", 
                        "content": "El perro corre rápidamente"
                    }
                ],
                "tensorFields": [],
                "mappings": {
                    "content": {
                        "type": "text_field",
                        "language": "es"
                    }
                }
            }
        )
        response.raise_for_status()
        
        # Test lexical search with language override
        search_response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/search",
            json={
                "q": "gatos",
                "searchMethod": "LEXICAL",
                "language": "es"
            }
        )
        search_response.raise_for_status()
        search_result = search_response.json()
        
        self.assertGreater(len(search_result["hits"]), 0)
        self.assertEqual(search_result["hits"][0]["_id"], "es1")

    def test_search_with_language_override_hybrid(self):
        """Test hybrid search with language override."""
        # Add documents with different languages
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/documents",
            json={
                "documents": [
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
                ],
                "tensorFields": ["title"],
                "mappings": {
                    "title": {
                        "type": "text_field",
                        "language": "fr"
                    },
                    "content": {
                        "type": "text_field",
                        "language": "fr"
                    }
                }
            }
        )
        response.raise_for_status()
        
        # Test hybrid search with language override
        search_response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/search",
            json={
                "q": "chat",
                "searchMethod": "HYBRID",
                "language": "fr"
            }
        )
        search_response.raise_for_status()
        search_result = search_response.json()
        
        self.assertGreater(len(search_result["hits"]), 0)

    def test_language_with_tensor_search_validation(self):
        """Test that language override is rejected for tensor search."""
        # Add some documents first
        response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/documents",
            json={
                "documents": [{"_id": "test1", "title": "Test content"}],
                "tensorFields": ["title"]
            }
        )
        response.raise_for_status()
        
        # Test that tensor search with language parameter raises error
        search_response = requests.post(
            f"{self.marqo_url}/indexes/{self.test_index_name}/search",
            json={
                "q": "test query",
                "searchMethod": "TENSOR",
                "language": "en"
            }
        )
        
        # Should return 422 validation error
        self.assertEqual(search_response.status_code, 422)
        error_result = search_response.json()
        error_msg = str(error_result)
        self.assertIn("language", error_msg)
        self.assertIn("TENSOR", error_msg)