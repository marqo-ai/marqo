import unittest
import base64
from io import BytesIO

from PIL import Image
import requests
from tests.marqo_test import MarqoTestCase, TestImageUrls


class TestBase64ImageSearchAPI(MarqoTestCase):
    """Test base64 image search functionality through the API."""

    structured_index_name = MarqoTestCase.random_index_name('structured_base64_index')
    unstructured_index_name = MarqoTestCase.random_index_name('unstructured_base64_index')
    image_model = 'open_clip/ViT-B-32/laion400m_e31'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create test base64 images for consistent testing
        cls.base64_images = cls._create_test_base64_images()

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": cls.image_model,
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image", "type": "image_pointer"},
                ],
                "tensorFields": ["image"],
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": cls.image_model,
                "treatUrlsAndPointersAsImages": True
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    @classmethod
    def _create_test_base64_images(cls):
        """Create test base64 images for consistent testing."""
        images = {}

        # Create different colored images for testing
        colors = [
            ('red_square', 'red'),
            ('blue_circle', 'blue'),
            ('green_triangle', 'green'),
            ('yellow_star', 'yellow')
        ]

        for name, color in colors:
            # Create a simple colored square image (20x20 pixels)
            img = Image.new('RGB', (20, 20), color=color)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            images[name] = {
                'data_url': f"data:image/png;base64,{base64_data}",
                'description': f"A {color} colored square"
            }

        return images

    @classmethod
    def _url_to_base64(cls, url: str):
        """Convert an image URL to base64 data URL format."""
        response = requests.get(url)
        response.raise_for_status()
        base64_data = base64.b64encode(response.content).decode('utf-8')
        # Determine content type from response headers or default to png
        content_type = response.headers.get('content-type', 'image/png')
        return f"data:{content_type};base64,{base64_data}"

    # def test_api_error_handling_invalid_base64(self):
    #     """Test API error handling for invalid base64 image data."""
    #     # Try to add document with invalid base64
    #     docs = [
    #         {
    #             "_id": "doc1",
    #             "image_field": "data:image/png;base64,invalid_base64_data!!!",
    #             "text_field": "Invalid base64 image"
    #         }
    #     ]
    #
    #     # This should handle the error gracefully during indexing
    #     # Depending on implementation, it might skip the field or return an error
    #     add_result = self.client.index(self.unstructured_index_name).add_documents(
    #         documents=docs,
    #         tensor_fields=["image_field", "text_field"]
    #     )
    #
    #     # Check that the API returns some response (exact behavior may vary)
    #     self.assertIn('items', add_result)

    def test_api_base64_images_rejected_in_add_documents(self):
        """Test that base64 images are properly rejected during document addition across all index types."""

        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        for index_type, index_name in index_configs:
            with self.subTest(index_type=index_type):
                # Test with data URL format base64 image - using same field names for both
                docs_with_data_url = [
                    {
                        "_id": "doc_with_base64_data_url",
                        "image": self.base64_images['red_square']['data_url'],
                        "title": "Document with base64 data URL"
                    }
                ]

                # Try to add document with base64 data URL - should fail
                if index_type == "unstructured":
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs_with_data_url,
                        tensor_fields=["image", "title"]
                    )
                else:
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs_with_data_url
                    )

                # Verify the request failed with appropriate error
                self.assertIn('items', add_result)
                self.assertEqual(len(add_result['items']), 1)
                item = add_result['items'][0]
                self.assertEqual(item['status'], 400)
                self.assertEqual(item['_id'], 'doc_with_base64_data_url')
                self.assertIn('base64 image data', item['message'].lower())
                self.assertIn('search queries', item['message'])

    def test_real_image_base64_search_all_methods_and_indexes(self):
        """Test base64 image search with real images (HIPPO_STATUE and COCO) across all index types and search methods."""
        # Convert real image URLs to base64 for search queries
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        # Define test parameters
        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        search_methods = [
            ("tensor", "TENSOR", None),
            ("hybrid_rrf", "HYBRID", {"retrievalMethod": "disjunction", "rankingMethod": "rrf"}),
            ("hybrid_tensor", "HYBRID", {"retrievalMethod": "tensor", "rankingMethod": "tensor"})
        ]

        for index_type, index_name in index_configs:
            with self.subTest(index_type=index_type):
                # Add documents with real image URLs
                docs = [
                    {
                        "_id": "hippo_doc",
                        "image": TestImageUrls.HIPPO_STATUE.value,
                        "title": "AI generated hippo statue"
                    },
                    {
                        "_id": "coco_doc",
                        "image": TestImageUrls.COCO.value,
                        "title": "COCO dataset image"
                    }
                ]

                # Add documents
                add_result = self.client.index(index_name).add_documents(
                    documents=docs,
                    tensor_fields=["image"] if index_type == "unstructured" else None
                )
                self.assertFalse(add_result['errors'])

                # Test each search method
                for search_name, search_method, hybrid_params in search_methods:
                    with self.subTest(search_method=search_name):
                        # Search with HIPPO_STATUE base64 image
                        search_params = {
                            "q": hippo_base64,
                            "search_method": search_method,
                        }

                        if hybrid_params:
                            search_params["hybrid_parameters"] = hybrid_params

                        search_result = self.client.index(index_name).search(**search_params)

                        # Verify results
                        self.assertIn('hits', search_result)
                        self.assertEqual(2, len(search_result['hits']))

                        # The hippo document should be the first hit since we're searching with hippo image
                        first_hit = search_result['hits'][0]
                        self.assertEqual('hippo_doc', first_hit['_id'])

                        # Verify score is 1 for the first hit
                        if search_name == 'hybrid_rrf':
                            score = first_hit['_tensor_score']
                        else:
                            score = first_hit['_score']
                        self.assertEqual(1.0, score, f"Score mismatch for {search_name} on {index_type} index")


if __name__ == '__main__':
    unittest.main()
