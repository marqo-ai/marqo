import unittest
import base64
from io import BytesIO

from PIL import Image
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.core.models.hybrid_parameters import HybridParameters


class TestBase64ImageSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create test base64 images for consistent testing
        cls.base64_images = cls._create_test_base64_images()

        # Create different index types for testing
        cls.unstructured_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            treat_urls_and_pointers_as_images=True
        )

        cls.structured_image_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch])
            ],
            tensor_fields=["text_field", "image_field", "title"]
        )

        # Create indexes
        cls.indexes = cls.create_indexes([cls.unstructured_image_index, cls.structured_image_index])
        cls.unstructured_image_index = cls.indexes[0]
        cls.structured_image_index = cls.indexes[1]

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
                'plain_base64': base64_data,
                'description': f"A {color} colored square"
            }
        
        return images

    def test_tensor_search_with_base64_data_url_unstructured(self):
        """Test tensor search with base64 data URL format images in unstructured index."""
        # Add documents with base64 images
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "A red square image"
            },
            {
                "_id": "doc2", 
                "image_field": self.base64_images['blue_circle']['data_url'],
                "text_field": "A blue circular image"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_image_index.name,
                docs=docs,
                tensor_fields=["image_field", "text_field"]
            )
        )
        
        # Search with base64 image
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_image_index.name,
            text=self.base64_images['red_square']['data_url'],
            search_method=SearchMethod.TENSOR,
            result_count=5
        )
        
        self.assertIn("hits", search_result)
        self.assertGreater(len(search_result["hits"]), 0)
        # The most similar should be the red square document
        self.assertEqual(search_result["hits"][0]["_id"], "doc1")

    def test_tensor_search_with_plain_base64_structured(self):
        """Test tensor search with plain base64 images in structured index."""
        # Add documents with plain base64 images
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['green_triangle']['plain_base64'],
                "text_field": "A green triangular shape",
                "title": "Green Triangle"
            },
            {
                "_id": "doc2",
                "image_field": self.base64_images['yellow_star']['plain_base64'], 
                "text_field": "A yellow star shape",
                "title": "Yellow Star"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.structured_image_index.name,
                docs=docs
            )
        )
        
        # Search with plain base64 image
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.structured_image_index.name,
            text=self.base64_images['green_triangle']['plain_base64'],
            search_method=SearchMethod.TENSOR,
            result_count=5
        )
        
        self.assertIn("hits", search_result)
        self.assertGreater(len(search_result["hits"]), 0)
        # The most similar should be the green triangle document
        self.assertEqual(search_result["hits"][0]["_id"], "doc1")

    def test_hybrid_search_with_base64_images_unstructured(self):
        """Test hybrid search with base64 images in unstructured index."""
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
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_image_index.name,
                docs=docs,
                tensor_fields=["image_field", "text_field"]
            )
        )
        
        # Test hybrid search with base64 image query
        hybrid_result = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_image_index.name,
            text=self.base64_images['red_square']['data_url'],
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod="disjunction",
                rankingMethod="rrf",
                verbose=True
            ),
            result_count=3
        )
        
        self.assertIn("hits", hybrid_result)
        self.assertGreater(len(hybrid_result["hits"]), 0)
        # Should find the matching red square image
        hit_ids = [hit["_id"] for hit in hybrid_result["hits"]]
        self.assertIn("doc1", hit_ids)

    def test_hybrid_search_with_text_and_base64_query(self):
        """Test hybrid search with both text and base64 image in query."""
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
                "text_field": "blue colored object"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_image_index.name,
                docs=docs,
                tensor_fields=["image_field", "text_field"]
            )
        )
        
        # Test tensor search with multi-modal query (hybrid search with dict queries is not supported)
        query = {
            "blue shape": 1.0,
            self.base64_images['blue_circle']['data_url']: 2.0
        }
        
        tensor_result = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_image_index.name,
            text=query,
            search_method=SearchMethod.TENSOR,
            result_count=3
        )
        
        self.assertIn("hits", tensor_result)
        self.assertGreater(len(tensor_result["hits"]), 0)
        # The blue circle document should rank highly due to image similarity + text match
        self.assertEqual(tensor_result["hits"][0]["_id"], "doc1")

    def test_tensor_search_base64_vs_url_similarity(self):
        """Test that base64 and URL versions of same image produce similar results."""
        # We'll use a known test image URL for comparison
        test_url = "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
        
        # Add documents with both URL and base64 versions
        docs = [
            {
                "_id": "url_doc",
                "image_field": test_url,
                "text_field": "hippo image from URL"
            },
            {
                "_id": "base64_doc", 
                "image_field": self.base64_images['red_square']['data_url'],
                "text_field": "red square from base64"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_image_index.name,
                docs=docs,
                tensor_fields=["image_field", "text_field"]
            )
        )
        
        # Search with base64 query
        base64_search = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_image_index.name,
            text=self.base64_images['red_square']['data_url'],
            search_method=SearchMethod.TENSOR,
            result_count=2
        )
        
        self.assertIn("hits", base64_search)
        self.assertGreater(len(base64_search["hits"]), 0)
        # Base64 document should be more similar to base64 query
        self.assertEqual(base64_search["hits"][0]["_id"], "base64_doc")

    def test_mixed_content_search_all_index_types(self):
        """Test search with mixed content (URLs, base64, text) across all index types."""
        for index in [self.unstructured_image_index, self.structured_image_index]:
            with self.subTest(index=index.name):
                docs = [
                    {
                        "_id": "base64_doc",
                        "image_field": self.base64_images['red_square']['data_url'],
                        "text_field": "red square base64 image"
                    },
                    {
                        "_id": "url_doc",
                        "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                        "text_field": "hippo URL image" 
                    },
                    {
                        "_id": "text_doc",
                        "text_field": "just text content about red squares"
                    }
                ]
                
                tensor_fields = ["image_field", "text_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=docs,
                        tensor_fields=tensor_fields
                    )
                )
                
                # Search with base64 image
                search_result = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text=self.base64_images['red_square']['data_url'],
                    search_method=SearchMethod.TENSOR,
                    result_count=3
                )
                
                self.assertIn("hits", search_result)
                self.assertGreater(len(search_result["hits"]), 0)
                # Should find documents, with base64 doc being most relevant
                hit_ids = [hit["_id"] for hit in search_result["hits"]]
                self.assertIn("base64_doc", hit_ids)

    def test_base64_image_highlights(self):
        """Test that base64 images are properly highlighted in search results."""
        docs = [
            {
                "_id": "doc1",
                "image_field": self.base64_images['yellow_star']['data_url'],
                "text_field": "yellow star image"
            }
        ]
        
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_image_index.name,
                docs=docs,
                tensor_fields=["image_field", "text_field"]
            )
        )
        
        # Search with highlights enabled
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_image_index.name,
            text=self.base64_images['yellow_star']['data_url'],
            search_method=SearchMethod.TENSOR,
            highlights=True,
            result_count=1
        )
        
        self.assertIn("hits", search_result)
        self.assertEqual(len(search_result["hits"]), 1)
        hit = search_result["hits"][0]
        
        # Check that highlights are present
        self.assertIn("_highlights", hit)
        self.assertGreater(len(hit["_highlights"]), 0)
        
        # The highlighted field should contain the base64 image
        highlighted_field = hit["_highlights"][0]
        self.assertIn("image_field", highlighted_field)
        self.assertEqual(highlighted_field["image_field"], self.base64_images['yellow_star']['data_url'])


if __name__ == '__main__':
    unittest.main()
