import base64
import unittest

import requests
from marqo.errors import MarqoWebError
from tests.marqo_test import MarqoTestCase, TestImageUrls


class TestBase64ImageSearch(MarqoTestCase):
    """Test base64 image search functionality through the API."""

    structured_index_name = MarqoTestCase.random_index_name('structured_base64_index')
    unstructured_index_name = MarqoTestCase.random_index_name('unstructured_base64_index')
    image_model = 'open_clip/ViT-B-32/laion2b_s34b_b79k'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

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
    def _url_to_base64(cls, url: str):
        """Convert an image URL to base64 data URL format."""
        response = requests.get(url)
        response.raise_for_status()
        base64_data = base64.b64encode(response.content).decode('utf-8')
        # Determine content type from response headers or default to png
        content_type = response.headers.get('content-type', 'image/png')
        return f"data:{content_type};base64,{base64_data}"

    def test_image_base64_search_all_methods_and_indexes(self):
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
            ("hybrid_tensor", "HYBRID", {"retrievalMethod": "tensor", "rankingMethod": "tensor"}),
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
                        "image": TestImageUrls.IMAGE1.value,
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
                        self.assertAlmostEqual(
                            1.0, score, places=3,
                            msg=f"Score mismatch for {search_name} on {index_type} index"
                        )

                        # Verify query not returned in base64 search response
                        self.assertEqual(
                            'data:image/[omitted]', search_result.get('query'),
                            'Base64 query should not be present in search response'
                        )

    def test_image_base64_search_large(self):
        """Test base64 image search with real images (HIPPO_STATUE and COCO) across all index types and search methods."""
        # Convert real image URLs to base64 for search queries
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_REALISTIC_LARGE.value)

        # Define test parameters
        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        search_methods = [
            ("tensor", "TENSOR", None),
            ("hybrid_rrf", "HYBRID", {"retrievalMethod": "disjunction", "rankingMethod": "rrf"}),
            ("hybrid_tensor", "HYBRID", {"retrievalMethod": "tensor", "rankingMethod": "tensor"}),
        ]

        # Add documents with real image URLs
        docs = [
            {
                "_id": "hippo_doc",
                "image": TestImageUrls.HIPPO_REALISTIC_LARGE.value,
                "title": "AI generated hippo statue"
            },
            {
                "_id": "coco_doc",
                "image": TestImageUrls.IMAGE1.value,
                "title": "COCO dataset image"
            }
        ]

        index_name = self.unstructured_index_name

        add_result = self.client.index(index_name).add_documents(
            documents=docs,
            tensor_fields=["image"]
        )
        self.assertFalse(add_result['errors'])

        search_params = {
            "q": hippo_base64,
            "search_method": 'TENSOR',
        }

        search_result = self.client.index(index_name).search(**search_params)

        # Verify results
        self.assertIn('hits', search_result)
        self.assertEqual(2, len(search_result['hits']))

        # The hippo document should be the first hit since we're searching with hippo image
        first_hit = search_result['hits'][0]
        self.assertEqual('hippo_doc', first_hit['_id'])

        # Verify score is 1 for the first hit
        score = first_hit['_score']
        self.assertAlmostEqual(1.0, score, places=3)

    def test_hybrid_search_with_base64_query_tensor_and_query_lexical(self):
        """Test hybrid search with base64 image in queryTensor and text in queryLexical across all index types."""
        # Convert real image URL to base64 for search query
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        for index_type, index_name in index_configs:
            with self.subTest(index_type=index_type):
                # Add documents with real image URLs and text
                docs = [
                    {
                        "_id": "hippo_statue_doc",
                        "image": TestImageUrls.HIPPO_STATUE.value,
                        "title": "AI generated hippo statue sculpture"
                    },
                    {
                        "_id": "coco_doc",
                        "image": TestImageUrls.IMAGE1.value,
                        "title": "COCO dataset image with various objects"
                    },
                    {
                        "_id": "text_only_hippo",
                        "title": "A document about hippo animals in the wild"
                    },
                    {
                        "_id": "text_only_statue",
                        "title": "Ancient statue sculpture art history"
                    }
                ]

                # Add documents
                if index_type == "unstructured":
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs,
                        tensor_fields=["image", "title"]
                    )
                else:
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs
                    )
                self.assertFalse(add_result['errors'])

                # Test 1: Basic base64 queryTensor with queryLexical
                with self.subTest(query_type="basic_base64"):
                    search_params = {
                        "search_method": "HYBRID",
                        "hybrid_parameters": {
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "queryTensor": hippo_base64,
                            "queryLexical": "statue sculpture"
                        }
                    }

                    search_result = self.client.index(index_name).search(**search_params)

                    # Verify results
                    self.assertIn('hits', search_result)
                    self.assertGreater(len(search_result['hits']), 0)

                    # The hippo statue document should be the first hit due to perfect image match
                    first_hit = search_result['hits'][0]
                    self.assertEqual(first_hit['_id'], 'hippo_statue_doc')

                    # Verify both tensor and lexical scores are present
                    self.assertIn('_tensor_score', first_hit)
                    self.assertIn('_lexical_score', first_hit)

                    # Tensor score should be 1.0 for perfect match
                    self.assertAlmostEqual(1.0, first_hit['_tensor_score'], places=3)

                # Test 2: Dict queryTensor with base64 (weight 1) and text (weight 0)
                with self.subTest(query_type="dict_base64_and_text"):
                    search_params = {
                        "search_method": "HYBRID",
                        "hybrid_parameters": {
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "queryTensor": {
                                hippo_base64: 1.0,
                                "elephant animal": 0.0  # Weight 0 means this won't affect results
                            },
                            "queryLexical": "sculpture art"
                        }
                    }

                    search_result = self.client.index(index_name).search(**search_params)

                    # Verify results
                    self.assertIn('hits', search_result)
                    self.assertGreater(len(search_result['hits']), 0)

                    # The hippo statue document should be the first hit due to perfect image match
                    first_hit = search_result['hits'][0]
                    self.assertEqual(first_hit['_id'], 'hippo_statue_doc')

                    # Tensor score should be 1.0 for perfect match (text with weight 0 shouldn't affect this)
                    self.assertAlmostEqual(1.0, first_hit['_tensor_score'], places=3)

    def test_tensor_search_with_base64_dict_query(self):
        """Test tensor search with dict query containing base64 image and text with weights across all index types."""
        # Convert real image URL to base64 for search query
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        for index_type, index_name in index_configs:
            with self.subTest(index_type=index_type):
                # Add documents with real image URLs and text
                docs = [
                    {
                        "_id": "hippo_statue_doc",
                        "image": TestImageUrls.HIPPO_STATUE.value,
                        "title": "AI generated hippo statue sculpture"
                    },
                    {
                        "_id": "coco_doc",
                        "image": TestImageUrls.IMAGE1.value,
                        "title": "COCO dataset image with various objects"
                    },
                    {
                        "_id": "text_only_hippo",
                        "title": "A document about hippo animals in the wild"
                    },
                    {
                        "_id": "text_only_elephant",
                        "title": "Elephant animal documentation and facts"
                    }
                ]

                # Add documents
                if index_type == "unstructured":
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs,
                        tensor_fields=["image", "title"]
                    )
                else:
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs
                    )
                self.assertFalse(add_result['errors'])

                # Test tensor search with dict query: base64 image (weight 0.8) and text (weight 0.2)
                search_params = {
                    "q": {
                        hippo_base64: 0.8,
                        "sculpture art": 0.2  # Lower weight for text
                    },
                    "search_method": "TENSOR"
                }

                search_result = self.client.index(index_name).search(**search_params)

                # Verify results
                self.assertIn('hits', search_result)
                self.assertGreater(len(search_result['hits']), 0)

                # The hippo statue document should still be the first hit due to strong image match
                first_hit = search_result['hits'][0]
                self.assertEqual(first_hit['_id'], 'hippo_statue_doc')

                # Score should still be high due to strong image component
                self.assertGreater(first_hit['_score'], 0.8)

                # Verify base64 string not returned
                self.assertEqual(
                    {
                        'data:image/[omitted]': 0.8,
                        "sculpture art": 0.2
                    },
                    search_result.get('query')
                )

    def test_invalid_base64_image_search_returns_400_error(self):
        """Test that searching with invalid base64 images returns 400 error for tensor and hybrid search."""
        # Invalid base64 data URL with malformed base64 content
        invalid_base64_data_url = "data:image/png;base64,invalid_base64_data!!!"

        index_configs = [
            ("unstructured", self.unstructured_index_name),
            ("structured", self.structured_index_name)
        ]

        search_methods = [
            ("tensor", "TENSOR"),
            ("hybrid_rrf", "HYBRID")
        ]

        for index_type, index_name in index_configs:
            with self.subTest(index_type=index_type):
                # Add some valid documents first
                docs = [
                    {
                        "_id": "test_doc",
                        "image": TestImageUrls.HIPPO_STATUE.value,
                        "title": "Test document for invalid base64 search"
                    }
                ]

                # Add documents
                if index_type == "unstructured":
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs,
                        tensor_fields=["image", "title"]
                    )
                else:
                    add_result = self.client.index(index_name).add_documents(
                        documents=docs
                    )
                self.assertFalse(add_result['errors'])

                # Test each search method with invalid base64
                for search_name, search_method in search_methods:
                    with self.subTest(search_method=search_name):
                        with self.assertRaises(MarqoWebError) as e:
                            self.client.index(index_name).search(
                                q=invalid_base64_data_url,
                                search_method=search_method
                            )
                        assert e.exception.status_code == 400

    def test_base64_images_rejected_in_add_documents(self):
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
                        "image": "data:image/png;base64,xxxyyyzzz",  # Invalid base64 data
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
                # Note: _id might be empty in error responses, so we check it exists as a key
                self.assertIn('_id', item)
                self.assertIn('base64 image data', item['message'].lower())
                self.assertIn('search queries', item['message'])


if __name__ == '__main__':
    unittest.main()
