import os
import unittest
import time
from unittest import mock
from marqo.config import Config
from marqo.vespa.vespa_client import VespaClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.core.models.marqo_index import Model, TextPreProcessing, ImagePreProcessing, DistanceMetric, \
    VectorNumericType, HnswConfig
from marqo.s2_inference import s2_inference


def pass_through_vectorise(*args, **kwargs):
    return s2_inference.vectorise(*args, **kwargs)


class TestPrefix(unittest.TestCase):
    def setUp(self):
        self.index_name = "my-test-index"
        self.device = "cpu"

        config_url = os.environ.get("VESPA_CONFIG_URL", "http://localhost:19071")
        document_url = os.environ.get("VESPA_DOCUMENT_URL", "http://localhost:8080")
        query_url = os.environ.get("VESPA_QUERY_URL", "http://localhost:8080")
        content_cluster_name = os.environ.get("VESPA_CONTENT_CLUSTER_NAME", "default")

        vespa_client = VespaClient(
            config_url=config_url,
            document_url=document_url,
            query_url=query_url,
            content_cluster_name=content_cluster_name,
        )

        self.marqo_config = Config(
            vespa_client=vespa_client,
            default_device=self.device
        )

        index_management = IndexManagement(vespa_client)

        marqo_index_request = UnstructuredMarqoIndexRequest(
            name=self.index_name,
            model=Model(
                name="my-custom-model",
                properties={
                    "name": "ViT-B-32-quickgelu",
                    "dimensions": 512,
                    "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                    "type": "open_clip",
                },
                custom=True,
            ),
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(),
            image_preprocessing=ImagePreProcessing(),
            distance_metric=DistanceMetric.Cosine,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(),
            treat_urls_and_pointers_as_images=True,
            filter_string_max_length=20,
        )

        try:
            index_management.delete_index_by_name(self.index_name)
        except:
            pass

        self.marqo_index = index_management.create_index(marqo_index_request)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": self.device})
        self.device_patcher.start()

    def tearDown(self):
        index_management = IndexManagement(self.marqo_config.vespa_client)
        try:
            index_management.delete_index_by_name(self.index_name)
        except:
            pass
        self.device_patcher.stop()

    def test_prefix_text_chunks(self):
        """Ensures that when adding documents with a prefix, each chunk has the prefix included in the vector,
        but the actual chunk text does not have the prefix."""
        
        # A) Add normal text document (1 chunk)
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True, device=self.device
        ))

        # B) Add same text document but WITH PREFIX (1 chunk) 
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True, 
            device=self.device, text_chunk_prefix="PREFIX: "
        ))
        
        # C) Add document with prefix built into text itself (1 chunk)
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_c", "text": "PREFIX: hello"}], auto_refresh=True,
            device=self.device
        ))

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.marqo_config, index_name=self.index_name, document_ids=["doc_a", "doc_b", "doc_c"],
            show_vectors=True  
        )["results"]
        retrieved_doc_a = res[0] 
        retrieved_doc_b = res[1]
        retrieved_doc_c = res[2]

        # Chunk content: For A) and B), should be exactly the same. C) is different.
        self.assertEqual(retrieved_doc_a["tensor_facets"][0]["text"], "hello")
        self.assertEqual(retrieved_doc_b["tensor_facets"][0]["text"], "hello") 
        self.assertEqual(retrieved_doc_c["tensor_facets"][0]["text"], "PREFIX: hello")

        # Chunk embedding: For B) and C), should be exactly the same. A) is different. 
        self.assertTrue(np.allclose(retrieved_doc_b["tensor_facets"][0]["_embedding"],
                                    retrieved_doc_c["tensor_facets"][0]["_embedding"]))
        self.assertFalse(np.allclose(retrieved_doc_a["tensor_facets"][0]["_embedding"], 
                                     retrieved_doc_c["tensor_facets"][0]["_embedding"]))

    @mock.patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise)
    def test_prefix_vectorise(self, mock_vectorise):
        """Ensures that vectorise is called on text with prefix"""
        
        # A) Add normal text document
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.device
        ))
        
        # B) Add same text document but WITH PREFIX (1 chunk)
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
            device=self.device, text_chunk_prefix="PREFIX: " 
        ))

        # Vectorise should be called twice, once with no prefix and once with prefix
        self.assertEqual(len(mock_vectorise.call_args_list), 2)
        args, kwargs = mock_vectorise.call_args_list[0]
        self.assertEqual(kwargs["content"], ["hello"])
        args, kwargs = mock_vectorise.call_args_list[1]  
        self.assertEqual(kwargs["content"], ["PREFIX: hello"])

    @mock.patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise) 
    def test_prefix_text_search(self, mock_vectorise):
        """Ensures that search query has prefix added to it for vectorisation."""
        
        # Add doc with no prefix
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.device
        ))
        # Add doc with prefix
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True, 
            device=self.device, text_chunk_prefix="PREFIX: "
        ))

        tensor_search.search(config=self.marqo_config, index_name=self.index_name, text="searching", device=self.device,
                             text_query_prefix="PREFIX: ")

        # Vectorise should be called 3 times - twice for add_documents, once for search 
        self.assertEqual(len(mock_vectorise.call_args_list), 3)
        args, kwargs = mock_vectorise.call_args_list[-1]
        self.assertEqual(kwargs["content"], ["PREFIX: searching"])

    def test_prefix_multimodal(self):
        """Ensures that vectorise is called on text list with prefixes, but image list without."""
        
        # Mocking rather than patching to capture call arguments
        s2_inference.vectorise = mock.MagicMock(side_effect=pass_through_vectorise)
        
        # Add a multimodal doc with a text and image field
        tensor_search.add_documents(config=self.marqo_config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{
                "_id": "doc_a", 
                "my-multimodal": {
                    "text": "hello",
                    "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
                }
            }], auto_refresh=True, 
            device=self.device, text_chunk_prefix="PREFIX: ",
            mappings={
                "my-multimodal": {
                    "type": "multimodal_combination",
                    "weights": {
                        "text": 0.5,
                        "image": 0.5
                    }
                }}
        ))

        # Check there were 2 vectorise calls:
        # 1st vectorise should be called with text with prefixes
        # 2nd vectorise should be called with just images 
        self.assertEqual(len(s2_inference.vectorise.call_args_list), 2)
        args, kwargs = s2_inference.vectorise.call_args_list[0]
        self.assertEqual(kwargs["content"], ["PREFIX: hello"])
        args, kwargs = s2_inference.vectorise.call_args_list[1]
        self.assertIsInstance(kwargs["content"][0], str) # uri
        self.assertEqual(kwargs["content"][0], "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png")

    @mock.patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise)
    def test_add_prefix_to_queries(self, mock_vectorise):
        """Ensures that prefix gets added to each query."""
        # Single text query (prefix added)
        queries = [SearchQuery(q="hello", text_query_prefix="PREFIX: ")]
        prefixed_queries = tensor_search.add_prefix_to_queries(queries)
        self.assertEqual(prefixed_queries[0].q, "PREFIX: hello")

        # Dict query (text has prefix, image does not) 
        queries = [SearchQuery(q={"text query": 0.5, "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5},
                               text_query_prefix="PREFIX: ")]
        prefixed_queries = tensor_search.add_prefix_to_queries(queries)
        self.assertEqual(prefixed_queries[0].q, {"PREFIX: text query": 0.5, "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5})

if __name__ == '__main__':
    unittest.main()