import copy
import os
import re

from marqo.tensor_search.models.add_docs_objects import AddDocsParams
import functools
import json
import math
import pprint
from unittest import mock
from unittest.mock import patch
from marqo.tensor_search.enums import EnvVars
from marqo.s2_inference import types, s2_inference
import PIL
from marqo.s2_inference.s2_inference import vectorise
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from marqo.tensor_search import enums
from marqo.errors import IndexNotFoundError, InvalidArgError, BadRequestError, InternalError
from marqo.tensor_search import tensor_search, index_meta_cache, backend
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import add_docs
from marqo.tensor_search.models.index_info import IndexInfo
import numpy as np
import unittest
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity

def pass_through_vectorise(*args, **kwargs):
    """Vectorise will behave as usual, but we will be able to see the call list
    via mock
    """
    return vectorise(*args, **kwargs)


class TestPrefix(MarqoTestCase):
    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"  # standard index created by setUp
        self.index_name_2 = "my-test-index-2"  # for tests that need custom index config
        self.index_name_3 = "my-test-index-3"  # No images

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_3)
        except IndexNotFoundError as s:
            pass

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

        # No default prefix!
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip",
                    }
                }
            }
        )

    def tearDown(self) -> None:
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.index_name_3 = "my-test-index-3"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_3)
        except IndexNotFoundError as s:
            pass

        self.device_patcher.stop()

    def test_prefix_text_chunks(self):
        """
        Ensures that when adding documents with a prefix, each chunk has the prefix included in the vector,
        but the actual chunk text does not have the prefix.
        """
        # A) Add normal text document (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
                device="cpu"
            )
        )
        
        # B) Add same text document but WITH PREFIX (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX: "
            )
        )

        # C) Add document with prefix built into text itself (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "doc_c", "text": "PREFIX: hello"}], auto_refresh=True,
                device="cpu"
            )
        )

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_1, document_ids=["doc_a", "doc_b", "doc_c"],
            show_vectors=True
        )["results"]
        retrieved_doc_a = res[0]
        retrieved_doc_b = res[1]
        retrieved_doc_c = res[2]

        # Chunk content: For A) and B), should be exactly the same. C) is different.
        assert retrieved_doc_a[TensorField.tensor_facets][0]["text"] == "hello"
        assert retrieved_doc_b[TensorField.tensor_facets][0]["text"] == "hello"
        assert retrieved_doc_c[TensorField.tensor_facets][0]["text"] == "PREFIX: hello"
        
        # Chunk embedding: For B) and C), should be exactly the same. A) is different.
        assert np.allclose(retrieved_doc_b[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_c[TensorField.tensor_facets][0]["_embedding"])
        assert not np.allclose(retrieved_doc_a[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_c[TensorField.tensor_facets][0]["_embedding"])

    def test_prefix_multiple_chunks(self):
        """
        Ensures that prefix gets added to each text chunk, not just the first one.
        """
        # Create index with custom model with NO DEFAULT PREFIX, split by WORD
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip",
                    },
                    "text_preprocessing": {
                        "split_method": "word",
                        "split_length": 1,
                        "split_overlap": 0
                    }
                }
            }
        )

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            # A) Add 1 document with 3 chunks, add PREFIX.
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_2, docs=[{"_id": "doc_a", "text": "one two three"}], auto_refresh=True,
                    device="cpu", text_chunk_prefix="PREFIX"
                )
            )

            # Vectorise should be called with prefixes included
            assert len(mock_vectorise.call_args_list) == 1
            args, kwargs = mock_vectorise.call_args_list[0]
            assert kwargs["content"] == ["PREFIXone", "PREFIXtwo", "PREFIXthree"]

            # B) Add 3 documents with 1 chunk, each with prefix built into text.
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_2, docs=[{"_id": "doc_b1", "text": "PREFIXone"}, {"_id": "doc_b2", "text": "PREFIXtwo"}, {"_id": "doc_b3", "text": "PREFIXthree"}], auto_refresh=True,
                    device="cpu"
                )
            )

        run()
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_2, document_ids=["doc_a", "doc_b1", "doc_b2", "doc_b3"],
            show_vectors=True
        )["results"]

        retrieved_doc_a = res[0]
        retrieved_doc_b1 = res[1]
        retrieved_doc_b2 = res[2]
        retrieved_doc_b3 = res[3]

        # The 3 chunks from A) should match the 3 from B)
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_b1[TensorField.tensor_facets][0]["_embedding"], atol=1e-5)
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][1]["_embedding"], retrieved_doc_b2[TensorField.tensor_facets][0]["_embedding"], atol=1e-5)
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][2]["_embedding"], retrieved_doc_b3[TensorField.tensor_facets][0]["_embedding"], atol=1e-5)
    
    def test_prefix_vectorise(self):
        """
        Ensures that vectorise is called on text with prefix
        """
        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            # A) Add normal text document
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
                    device="cpu"
                )
            )
            
            # B) Add same text document but WITH PREFIX (1 chunk)
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
                    device="cpu", text_chunk_prefix="PREFIX: "
                )
            )
        
        run()

        assert len(mock_vectorise.call_args_list) == 2
        args, kwargs = mock_vectorise.call_args_list[0]
        assert kwargs["content"] == ["hello"]
        args, kwargs = mock_vectorise.call_args_list[1]
        assert kwargs["content"] == ["PREFIX: hello"]

    def test_prefix_not_on_images(self):
        """
        Ensures that prefix does not get added to image, unless treat_urls_as_images is False.
        """
        # Create index with custom model with NO DEFAULT PREFIX, treat_urls_and_pointers_as_images is False
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": False
                }
            }
        )

        # Add a doc with an image URL (WITH PREFIX)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_a", "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX: "
            )
        )

        # Add a doc with an image URL (prefix built into text)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_b", "image": "PREFIX: https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}], auto_refresh=True,
                device="cpu"
            )
        )

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_2, document_ids=["doc_a", "doc_b"],
            show_vectors=True
        )["results"]
        retrieved_doc_a = res[0]
        retrieved_doc_b = res[1]

        # Chunk content: Should be different
        assert retrieved_doc_a[TensorField.tensor_facets][0]["image"] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
        assert retrieved_doc_b[TensorField.tensor_facets][0]["image"] == "PREFIX: https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"

        # Chunk embedding: Should be the same
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_b[TensorField.tensor_facets][0]["_embedding"])

        # Create index with custom model with NO DEFAULT PREFIX, treat_urls_and_pointers_as_images is False
        tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        # Add a doc with an image URL (WITH PREFIX)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_c", "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX: "
            )
        )

        # Add a doc with an image URL (NO PREFIX)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_d", "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}], auto_refresh=True,
                device="cpu"
            )
        )

        # Get the documents
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_2, document_ids=["doc_c", "doc_d"],
            show_vectors=True
        )["results"]
        retrieved_doc_c = res[0]
        retrieved_doc_d = res[1]

        # Chunk content should be the same
        assert retrieved_doc_c[TensorField.tensor_facets][0]["image"] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
        assert retrieved_doc_d[TensorField.tensor_facets][0]["image"] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"

        # Chunk embedding should be the same (because prefixes do not affect images)
        assert np.allclose(retrieved_doc_c[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_d[TensorField.tensor_facets][0]["_embedding"])
        
    def test_prefix_multimodal(self):
        """
        Ensures that vectorise is called on text list with prefixes, but image list without.
        """

        # Create index with custom model with NO DEFAULT PREFIX, treat_urls_as_images is True
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            # Add a multimodal doc with a text and image field
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_2, docs=[{
                        "_id": "doc_a", 
                        "my-multimodal": {
                            "text": "hello", 
                            "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
                        }
                    }], auto_refresh=True,
                    device="cpu", text_chunk_prefix="PREFIX: ", 
                    mappings={
                    "my-multimodal": {
                        "type": "multimodal_combination",
                        "weights": {
                            "text": 0.5,
                            "image": 0.5
                        }
                    }}
                )
            )
        
        run()

        # Check there were 2 vectorise calls:
        # 1st vectorise should be called with text with prefixes
        # 2nd vectorise should be called with just images

        assert len(mock_vectorise.call_args_list) == 2
        args, kwargs = mock_vectorise.call_args_list[0]
        assert kwargs["content"] == ["PREFIX: hello"]
        args, kwargs = mock_vectorise.call_args_list[1]
        assert isinstance(kwargs["content"][0], PIL.PngImagePlugin.PngImageFile)
        
        # Get document, content chunks should not have prefix
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_2, document_ids=["doc_a"],
            show_vectors=True
        )["results"]
        retrieved_doc_a = res[0]

        assert retrieved_doc_a[TensorField.tensor_facets][0]["my-multimodal"] == json.dumps({
            "text": "hello", 
            "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"
        })
    
    def test_prefix_text_search(self):
        """
        Ensures that search query has prefix added to it for vectorisation.
        Use pass through vectorise.
        """
        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            # Add doc with no prefix
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
                device="cpu"
                )
            )
            # Add doc with prefix
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX: "
                )
            )

            res = tensor_search.search(config=self.config, index_name=self.index_name_1, text="searching", device="cpu", text_query_prefix="PREFIX: ")
            assert res["hits"][0]["_id"] == "doc_b"     # Because doc b had the prefix added.
            assert res["query"] == "searching"  # No prefix in the query itself
        
        run()

        assert len(mock_vectorise.call_args_list) == 3
        args, kwargs = mock_vectorise.call_args_list[-1]
        assert kwargs["content"] == ["PREFIX: searching"]
    
    def test_prefix_image_search(self):
        """
        Ensures that an image search query has prefix added to it for vectorisation.
        """
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            # Add image with no prefix
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_a", "image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}], auto_refresh=True,
                device="cpu"
                )
            )
            # Add doc
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_b", "text": "red herring marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX: "
                )
            )

            res = tensor_search.search(config=self.config, index_name=self.index_name_2, text="https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png", device="cpu", text_query_prefix="PREFIX: ")
            assert res["hits"][0]["_id"] == "doc_a"
            assert res["query"] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"  # No prefix in the query itself
        
        run()

        assert len(mock_vectorise.call_args_list) == 3
        args, kwargs = mock_vectorise.call_args_list[-1]
        assert kwargs["content"][0] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"


    def test_prefix_dict_search(self):
        """
        Ensures that dict search query has prefix added to each for vectorisation.
        Use pass through vectorise.
        """
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            query_dict = {
                "text query": 0.5,
                "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5,
            }
            res = tensor_search.search(config=self.config, index_name=self.index_name_2, 
                                       text=query_dict, device="cpu", text_query_prefix="PREFIX: ")
            assert res["query"] == query_dict  # No prefix in the query itself
        
        run()

        assert len(mock_vectorise.call_args_list) == 2
        args, kwargs = mock_vectorise.call_args_list[0]
        assert kwargs["content"][0] == "PREFIX: text query"
        args, kwargs = mock_vectorise.call_args_list[1]
        assert kwargs["content"][0] == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"

        
    def test_add_prefix_to_queries(self):
        """
        Ensures that prefix gets added to each query.
        """

        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_2, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_3, 
            index_settings={
                "index_defaults": {
                    "model": "my-custom-model",
                    "model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip"
                    },
                    "treat_urls_and_pointers_as_images": False
                }
            }
        )

        # Single text query (prefix added)
        queries = [
            BulkSearchQueryEntity(
                q="hello",
                textQueryPrefix="PREFIX: ",
                index=self.index_name_2
            )
        ]
        prefixed_queries = tensor_search.add_prefix_to_queries(self.config, queries)
        assert prefixed_queries[0].q == "PREFIX: hello"

        # Single image query (no prefix added)
        queries = [
            BulkSearchQueryEntity(
                q="https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                textQueryPrefix="PREFIX: ",
                index=self.index_name_2
            )
        ]
        prefixed_queries = tensor_search.add_prefix_to_queries(self.config, queries)
        assert prefixed_queries[0].q == "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"

        # Dict query (text has prefix, image does not)
        queries = [
            BulkSearchQueryEntity(
                q={
                    "text query": 0.5,
                    "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5,
                },
                textQueryPrefix="PREFIX: ",
                index=self.index_name_2
            )
        ]
        prefixed_queries = tensor_search.add_prefix_to_queries(self.config, queries)
        assert prefixed_queries[0].q == {
            "PREFIX: text query": 0.5,
            "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5,
        }

        # Single image but no image index (prefix added)
        queries = [
            BulkSearchQueryEntity(
                q="https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png",
                textQueryPrefix="PREFIX: ",
                index=self.index_name_3
            )
        ]
        prefixed_queries = tensor_search.add_prefix_to_queries(self.config, queries)
        assert prefixed_queries[0].q == "PREFIX: https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"

        # Dict query but no image index (text and index have prefix)
        queries = [
            BulkSearchQueryEntity(
                q={
                    "text query": 0.5,
                    "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5,
                },
                textQueryPrefix="PREFIX: ",
                index=self.index_name_3
            )
        ]
        prefixed_queries = tensor_search.add_prefix_to_queries(self.config, queries)
        assert prefixed_queries[0].q == {
            "PREFIX: text query": 0.5,
            "PREFIX: https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5,
        }
    

    