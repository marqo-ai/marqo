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
import requests
import pytest
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
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
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
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
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
        
        # A) Add 1 document with 3 chunks, add PREFIX.
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_a", "text": "one two three"}], auto_refresh=True,
                device="cpu", text_chunk_prefix="PREFIX"
            )
        )

        # B) Add 3 documents with 1 chunk, each with prefix built into text.
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_2, docs=[{"_id": "doc_b1", "text": "PREFIXone"}, {"_id": "doc_b2", "text": "PREFIXtwo"}, {"_id": "doc_b3", "text": "PREFIXthree"}], auto_refresh=True,
                device="cpu"
            )
        )

        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name_2, document_ids=["doc_a", "doc_b1", "doc_b2", "doc_b3"],
            show_vectors=True
        )["results"]

        retrieved_doc_a = res[0]
        retrieved_doc_b1 = res[1]
        retrieved_doc_b2 = res[2]
        retrieved_doc_b3 = res[3]

        # The 3 chunks from A) should match the 3 from B)
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][0]["_embedding"], retrieved_doc_b1[TensorField.tensor_facets][0]["_embedding"])
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][1]["_embedding"], retrieved_doc_b2[TensorField.tensor_facets][0]["_embedding"])
        assert np.allclose(retrieved_doc_a[TensorField.tensor_facets][2]["_embedding"], retrieved_doc_b3[TensorField.tensor_facets][0]["_embedding"])
    
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
        
        run()

        assert len(mock_vectorise.call_args_list) == 3
        args, kwargs = mock_vectorise.call_args_list[-1]
        assert kwargs["content"] == ["PREFIX: searching"]



    def test_prefix_dict_search(self):
        """
        Ensures that dict search query has prefix added to each for vectorisation.
        Use pass through vectorise.
        """
        pass


        