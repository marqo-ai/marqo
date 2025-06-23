import unittest
import base64
from io import BytesIO

from PIL import Image
from tests.marqo_test import MarqoTestCase


class TestBase64ImageSearchAPI(MarqoTestCase):
    """Test base64 image search functionality through the API."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        
        # Create test base64 images
        cls.base64_images = cls._create_test_base64_images()
        
        # Create test indexes
        cls.unstructured_index_name = "test-base64-unstructured"
        cls.structured_index_name = "test-base64-structured" 
        
        # Delete indexes if they exist
        try:
            cls.client.delete_index(cls.unstructured_index_name)
        except:
            pass  # Index doesn't exist, which is fine
        
        try:
            cls.client.delete_index(cls.structured_index_name)
        except:
            pass  # Index doesn't exist, which is fine
        
        # Create unstructured index
        cls.client.create_index(
            index_name=cls.unstructured_index_name,
            settings_dict={
                "model": "open_clip/ViT-B-32/laion400m_e31",
                "treatUrlsAndPointersAsImages": True,
                "type": "unstructured"
            }
        )
        
        # Create structured index  
        cls.client.create_index(
            index_name=cls.structured_index_name,
            settings_dict={
                "type": "structured",
                "model": "open_clip/ViT-B-32/laion400m_e31",
                "allFields": [
                    {"name": "text_field", "type": "text", "features": ["lexical_search"]},
                    {"name": "image_field", "type": "image_pointer"},
                    {"name": "title", "type": "text", "features": ["lexical_search"]}
                ],
                "tensorFields": ["text_field", "image_field", "title"]
            }
        )

    @classmethod
    def _create_test_base64_images(cls):
        """Create test base64 images for API testing."""
        images = {}
        
        colors = [('red_square', 'red'), ('blue_circle', 'blue'), ('green_triangle', 'green')]
        
        for name, color in colors:
            # Create a small colored image (15x15 pixels)
            img = Image.new('RGB', (15, 15), color=color)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            images[name] = {
                'data_url': f"data:image/png;base64,{base64_data}",
                'plain_base64': base64_data
            }
        
        return images

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up test indexes
        try:
            cls.client.delete_index(cls.unstructured_index_name)
        except:
            pass
        
        try:
            cls.client.delete_index(cls.structured_index_name)
        except:
            pass
        
        super().tearDownClass()

    def setUp(self) -> None:
        # Clear all documents from indexes before each test
        try:
            self.client.index(self.unstructured_index_name).delete_documents()
        except:
            pass
        
        try:
            self.client.index(self.structured_index_name).delete_documents()  
        except:
            pass

    def test_api_tensor_search_with_base64_data_url(self):
        """Test tensor search with base64 data URL through API."""
        # Add documents with base64 images
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "Red square image"
            },
            {
                "_id": "doc2", 
                "image_field": self.base64_images['blue_circle']['data_url'],
                "text_field": "Blue circle image"
            }
        ]
        
        # Add documents through API
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        self.assertIn('items', add_result)
        for item in add_result['items']:
            self.assertEqual(item['status'], 200)
        
        # Search with base64 image query
        search_result = self.client.index(self.unstructured_index_name).search(
            q=self.base64_images['red_square']['data_url'],
            search_method="TENSOR",
            limit=5
        )
        
        self.assertIn('hits', search_result)
        self.assertGreater(len(search_result['hits']), 0)
        # Should find the red square document as most similar
        hit_ids = [hit['_id'] for hit in search_result['hits']]
        self.assertIn('data_url_doc', hit_ids)

    def test_api_hybrid_search_with_base64_images(self):
        """Test hybrid search with base64 images through API."""
        # Add documents with mixed content
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "red square geometric shape"
            },
            {
                "_id": "doc2",
                "image_field": self.base64_images['blue_circle']['data_url'],
                "text_field": "blue circle round shape"
            },
            {
                "_id": "doc3",
                "text_field": "red color description without image"
            }
        ]
        
        # Add documents
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        self.assertIn('items', add_result)
        for item in add_result['items']:
            self.assertEqual(item['status'], 200)
        
        # Test hybrid search with base64 image
        hybrid_result = self.client.index(self.unstructured_index_name).search(
            q=self.base64_images['red_square']['data_url'],
            search_method="HYBRID",
            hybrid_parameters={
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf"
            },
            limit=3
        )
        
        self.assertIn('hits', hybrid_result)
        self.assertGreater(len(hybrid_result['hits']), 0)
        # Should find documents with red square being highly ranked
        hit_ids = [hit['_id'] for hit in hybrid_result['hits']]
        self.assertIn('doc1', hit_ids)

    def test_api_multimodal_query_with_base64(self):
        """Test multimodal query with both text and base64 image through API."""
        # Add documents
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['blue_circle']['data_url'],
                "text_field": "blue circle shape"
            },
            {
                "_id": "doc2",
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "red square shape" 
            },
            {
                "_id": "doc3",
                "text_field": "blue colored object description"
            }
        ]
        
        # Add documents
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        self.assertIn('items', add_result)
        
        # Test multimodal query with weighted terms
        query = {
            "blue shape": 1.0,
            self.base64_images['blue_circle']['data_url']: 2.0
        }
        
        search_result = self.client.index(self.unstructured_index_name).search(
            q=query,
            search_method="TENSOR",
            limit=3
        )
        
        self.assertIn('hits', search_result)
        self.assertGreater(len(search_result['hits']), 0)
        # Blue circle document should rank highest due to image similarity + text match
        self.assertEqual(search_result['hits'][0]['_id'], 'doc1')

    def test_api_base64_image_with_highlights(self):
        """Test that base64 images work properly with highlights through API."""
        # Add document with base64 image
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['green_triangle']['data_url'],
                "text_field": "green triangle image"
            }
        ]
        
        # Add documents
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        self.assertIn('items', add_result)
        self.assertEqual(add_result['items'][0]['status'], 200)
        
        # Search with highlights
        search_result = self.client.index(self.unstructured_index_name).search(
            q=self.base64_images['green_triangle']['data_url'],
            search_method="TENSOR",
            highlights=True,
            limit=1
        )
        
        self.assertIn('hits', search_result)
        self.assertEqual(len(search_result['hits']), 1)
        hit = search_result['hits'][0]
        
        # Check highlights
        self.assertIn('_highlights', hit)
        self.assertGreater(len(hit['_highlights']), 0)
        
        # The highlighted field should contain the base64 image
        highlighted_field = hit['_highlights'][0]
        self.assertIn('image_field', highlighted_field)
        self.assertEqual(highlighted_field['image_field'], self.base64_images['green_triangle']['data_url'])

    def test_api_error_handling_invalid_base64(self):
        """Test API error handling for invalid base64 image data."""
        # Try to add document with invalid base64
        docs = [
            {
                "_id": "doc1",
                "image_field": "data:image/png;base64,invalid_base64_data!!!",
                "text_field": "Invalid base64 image"
            }
        ]
        
        # This should handle the error gracefully during indexing
        # Depending on implementation, it might skip the field or return an error
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        # Check that the API returns some response (exact behavior may vary)
        self.assertIn('items', add_result)

    def test_api_mixed_image_formats(self):
        """Test API with mixed image formats (URL, base64 data URL, plain base64)."""
        # Add documents with different image formats
        docs = [
            {
                "_id": "url_doc",
                "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                "text_field": "hippo from URL"
            },
            {
                "_id": "data_url_doc",
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "red square from data URL"
            },
            {
                "_id": "base64_doc", 
                "image_field": self.base64_images['blue_circle']['plain_base64'],
                "text_field": "blue circle from base64"
            }
        ]
        
        # Add documents
        add_result = self.client.index(self.unstructured_index_name).add_documents(
            documents=docs,
            tensor_fields=["image_field", "text_field"]
        )
        
        self.assertIn('items', add_result)
        for item in add_result['items']:
            self.assertEqual(item['status'], 200)
        
        # Search with base64 data URL should find the matching document
        search_result = self.client.index(self.unstructured_index_name).search(
            q=self.base64_images['red_square']['data_url'],
            search_method="TENSOR",
            limit=3
        )
        
        self.assertIn('hits', search_result)
        self.assertGreater(len(search_result['hits']), 0)
        # Should find the red square document as most similar
        hit_ids = [hit['_id'] for hit in search_result['hits']]
        self.assertIn('data_url_doc', hit_ids)


if __name__ == '__main__':
    unittest.main() 