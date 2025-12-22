import base64
import unittest

import requests

from marqo.api.exceptions import InvalidArgError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase, TestImageUrls


class TestBase64ImageSearch(MarqoTestCase):
    """Integration tests for base64 image search functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create unstructured index for base64 tests
        cls.unstructured_base64_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion2b_s34b_b79k'),
            treat_urls_and_pointers_as_images=True
        )

        # Create structured index for base64 tests
        cls.structured_base64_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion2b_s34b_b79k'),
            fields=[
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
                FieldRequest(name="image", type=FieldType.ImagePointer),
            ],
            tensor_fields=["image"]
        )

        # Create unstructured index with marqo_version 2.12 for base64 tests
        cls.unstructured_base64_v212_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion2b_s34b_b79k'),
            treat_urls_and_pointers_as_images=True,
            marqo_version='2.12.0'
        )

        cls.indexes = cls.create_indexes([
            cls.unstructured_base64_index,
            cls.structured_base64_index,
            cls.unstructured_base64_v212_index
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_base64_index = cls.indexes[0]
        cls.structured_base64_index = cls.indexes[1]
        cls.unstructured_base64_v212_index = cls.indexes[2]

    def setUp(self) -> None:
        super().setUp()

    @classmethod
    def _url_to_base64(cls, url: str):
        """Convert an image URL to base64 data URL format."""
        response = requests.get(url)
        response.raise_for_status()
        base64_data = base64.b64encode(response.content).decode('utf-8')
        # Determine content type from response headers or default to png
        content_type = response.headers.get('content-type', 'image/png')
        return f"data:{content_type};base64,{base64_data}"

    def test_real_image_base64_search_all_methods_and_indexes(self):
        """Test base64 image search with real images across all index types and search methods."""
        # Convert real image URLs to base64 for search queries
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        # Define test parameters
        index_configs = [
            ("unstructured", self.unstructured_base64_index),
            ("structured", self.structured_base64_index),
            ("unstructured_v212", self.unstructured_base64_v212_index)
        ]

        search_methods = [
            ("tensor", SearchMethod.TENSOR, None),
            ("hybrid_rrf", SearchMethod.HYBRID, {"retrievalMethod": "disjunction", "rankingMethod": "rrf"}),
            ("hybrid_tensor", SearchMethod.HYBRID, {"retrievalMethod": "tensor", "rankingMethod": "tensor"}),
        ]

        for index_type, marqo_index in index_configs:
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
                        "image": TestImageUrls.IMAGE0.value,
                        "title": "Image search guide image"
                    }
                ]

                # Add documents
                if index_type in ["unstructured", "unstructured_v212"]:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs,
                            tensor_fields=["image"]
                        )
                    )
                else:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs
                        )
                    )

                # Test each search method
                for search_name, search_method, hybrid_params in search_methods:
                    with self.subTest(search_method=search_name):
                        # Search with HIPPO_STATUE base64 image
                        if hybrid_params:
                            hybrid_parameters = HybridParameters(**hybrid_params)
                        else:
                            hybrid_parameters = None

                        search_result = tensor_search.search(
                            config=self.config,
                            index_name=marqo_index.name,
                            text=hippo_base64,
                            search_method=search_method,
                            result_count=10,
                            hybrid_parameters=hybrid_parameters
                        )

                        print(search_result)

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

    def test_hybrid_search_with_base64_query_tensor_and_query_lexical(self):
        """Test hybrid search with base64 image in queryTensor and text in queryLexical across all index types."""
        # Convert real image URL to base64 for search query
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        index_configs = [
            ("unstructured", self.unstructured_base64_index),
            ("structured", self.structured_base64_index),
            ("unstructured_v212", self.unstructured_base64_v212_index)
        ]

        for index_type, marqo_index in index_configs:
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
                        "image": TestImageUrls.IMAGE0.value,
                        "title": "Image search guide image with various objects"
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
                if index_type in ["unstructured", "unstructured_v212"]:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs,
                            tensor_fields=["image", "title"]
                        )
                    )
                else:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs
                        )
                    )

                # Test 1: Basic base64 queryTensor with queryLexical
                with self.subTest(query_type="basic_base64"):
                    from marqo.core.models.hybrid_parameters import HybridParameters

                    hybrid_params = HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        queryTensor=hippo_base64,
                        queryLexical="statue sculpture"
                    )

                    search_result = tensor_search.search(
                        config=self.config,
                        index_name=marqo_index.name,
                        text=None,  # No main query text for hybrid with separate queryTensor/queryLexical
                        search_method=SearchMethod.HYBRID,
                        result_count=10,
                        hybrid_parameters=hybrid_params
                    )

                    # Verify results
                    self.assertIn('hits', search_result)
                    self.assertGreater(len(search_result['hits']), 0)

                    # The hippo statue document should be the first hit due to perfect image match
                    first_hit = search_result['hits'][0]
                    self.assertEqual(first_hit['_id'], 'hippo_statue_doc')

                    # Tensor score should be 1.0 for perfect match
                    self.assertAlmostEqual(1.0, first_hit['_tensor_score'], places=3)

                    # Verify query field is None since text=None was passed
                    self.assertIsNone(search_result.get('query'))

                # Test 2: Dict queryTensor with base64 (weight 1) and text (weight 0)
                with self.subTest(query_type="dict_base64_and_text"):
                    hybrid_params = HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        queryTensor={
                            hippo_base64: 1.0,
                            "elephant animal": 0.0  # Weight 0 means this won't affect results
                        },
                        queryLexical="sculpture art"
                    )

                    search_result = tensor_search.search(
                        config=self.config,
                        index_name=marqo_index.name,
                        text=None,
                        search_method=SearchMethod.HYBRID,
                        result_count=10,
                        hybrid_parameters=hybrid_params
                    )

                    # Verify results
                    self.assertIn('hits', search_result)
                    self.assertGreater(len(search_result['hits']), 0)

                    # The hippo statue document should be the first hit due to perfect image match
                    first_hit = search_result['hits'][0]
                    self.assertEqual(first_hit['_id'], 'hippo_statue_doc')

                    # Tensor score should be 1.0 for perfect match (text with weight 0 shouldn't affect this)
                    self.assertAlmostEqual(1.0, first_hit['_tensor_score'], places=3)

                    # Verify query field is None since text=None was passed
                    self.assertIsNone(search_result.get('query'))

    def test_tensor_search_with_base64_dict_query(self):
        """Test tensor search with dict query containing base64 image and text with weights across all index types."""
        # Convert real image URL to base64 for search query
        hippo_base64 = self._url_to_base64(TestImageUrls.HIPPO_STATUE.value)

        index_configs = [
            ("unstructured", self.unstructured_base64_index),
            ("structured", self.structured_base64_index),
            ("unstructured_v212", self.unstructured_base64_v212_index)
        ]

        for index_type, marqo_index in index_configs:
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
                        "image": TestImageUrls.IMAGE0.value,
                        "title": "Image search guide image with various objects"
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
                if index_type in ["unstructured", "unstructured_v212"]:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs,
                            tensor_fields=["image", "title"]
                        )
                    )
                else:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs
                        )
                    )

                # Test tensor search with dict query: base64 image (weight 0.8) and text (weight 0.2)
                with self.subTest(query_type="dict_mixed_weights"):
                    search_result = tensor_search.search(
                        config=self.config,
                        index_name=marqo_index.name,
                        text={
                            hippo_base64: 0.8,
                            "sculpture art": 0.2  # Lower weight for text
                        },
                        search_method=SearchMethod.TENSOR,
                        result_count=10
                    )

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

    def test_invalid_base64_image_search_raises_error(self):
        """Test that searching with invalid base64 images raises InvalidArgError for tensor and hybrid search."""
        # Invalid base64 data URL with malformed base64 content
        invalid_base64_data_url = "data:image/png;base64,invalid_base64_data!!!"

        index_configs = [
            ("unstructured", self.unstructured_base64_index),
            ("structured", self.structured_base64_index),
            ("unstructured_v212", self.unstructured_base64_v212_index)
        ]

        search_methods = [
            ("tensor", SearchMethod.TENSOR),
            ("hybrid_rrf", SearchMethod.HYBRID)
        ]

        for index_type, marqo_index in index_configs:
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
                if index_type in ["unstructured", "unstructured_v212"]:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs,
                            tensor_fields=["image", "title"]
                        )
                    )
                else:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs
                        )
                    )

                # Test each search method with invalid base64
                for search_name, search_method in search_methods:
                    with self.subTest(search_method=search_name):
                        # Invalid base64 should raise an exception (could be InternalError or binascii.Error)
                        with self.assertRaises(InvalidArgError):
                            tensor_search.search(
                                config=self.config,
                                index_name=marqo_index.name,
                                text=invalid_base64_data_url,
                                search_method=search_method,
                                result_count=10
                            )

    def test_base64_images_rejected_in_add_documents(self):
        """Test that base64 images are properly rejected during document addition across all index types."""
        index_configs = [
            ("unstructured", self.unstructured_base64_index),
            ("structured", self.structured_base64_index),
            ("unstructured_v212", self.unstructured_base64_v212_index)
        ]

        for index_type, marqo_index in index_configs:
            with self.subTest(index_type=index_type):
                # Test with data URL format base64 image
                docs_with_data_url = [
                    {
                        "_id": "doc_with_base64_data_url",
                        "image": "data:image/png;base64,xxxyyyzzz",  # Invalid base64 data
                        "title": "Document with base64 data URL"
                    }
                ]

                # Try to add document with base64 data URL - should return errors in response
                if index_type in ["unstructured", "unstructured_v212"]:
                    result = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs_with_data_url,
                            tensor_fields=["image", "title"]
                        )
                    )
                else:
                    result = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=marqo_index.name, docs=docs_with_data_url
                        )
                    )

                # Verify the request failed with appropriate error
                self.assertTrue(result.errors)
                self.assertIn('base64 image data', result.items[0].message)
                self.assertIn('search queries', result.items[0].message)


if __name__ == '__main__':
    unittest.main()
