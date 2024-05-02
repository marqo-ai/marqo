import unittest

import numpy as np
from unittest import mock
from unittest.mock import Mock, patch
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity
from marqo.core.models.marqo_index import UnstructuredMarqoIndex, StructuredMarqoIndex
from marqo.s2_inference import s2_inference
from tests.marqo_test import MarqoTestCase


def pass_through_vectorise(*args, **kwargs):
    return s2_inference.vectorise(*args, **kwargs)


class TestPrefix(MarqoTestCase):
    #mock_vectorise = unittest.mock.MagicMock()
    #mock_vectorise.side_effect = pass_through_vectorise

    def setUp(self) -> None:
        self.index_name = self.random_index_name()
        self.index_request = self.unstructured_marqo_index_request(name=self.index_name, treat_urls_and_pointers_as_images=True)
        self.marqo_index = self.create_indexes([self.index_request])[0]

    def test_prefix_text_chunks(self):
        """Ensures that when adding documents with a prefix, each chunk has the prefix included in the vector,
        but the actual chunk text does not have the prefix."""

        # A) Add normal text document (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, tensor_fields=["text"]
        ))

        # B) Add same text document but WITH PREFIX (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, text_chunk_prefix="PREFIX: ", tensor_fields=["text"]
        ))

        # C) Add document with prefix built into text itself (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_c", "text": "PREFIX: hello"}], auto_refresh=True,
            device=self.config.default_device, tensor_fields=["text"]
        ))

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name, document_ids=["doc_a", "doc_b", "doc_c"],
            show_vectors=True
        )["results"]
        retrieved_doc_a = res[0]
        retrieved_doc_b = res[1]
        retrieved_doc_c = res[2]

        # Chunk content: For A) and B), should be exactly the same. C) is different.
        self.assertEqual(retrieved_doc_a["text"], "hello")
        self.assertEqual(retrieved_doc_b["text"], "hello")
        self.assertEqual(retrieved_doc_c["text"], "PREFIX: hello")

        # Chunk embedding: For B) and C), should be exactly the same. A) is different.
        self.assertTrue(np.allclose(retrieved_doc_b["_tensor_facets"][0]["_embedding"],
                                    retrieved_doc_c["_tensor_facets"][0]["_embedding"]))
        self.assertFalse(np.allclose(retrieved_doc_a["_tensor_facets"][0]["_embedding"],
                                     retrieved_doc_c["_tensor_facets"][0]["_embedding"]))

    def test_prefix_multimodal(self):
        """Ensures that vectorise is called on text list with prefixes, but image list without."""

        # Add a multimodal doc with a text and image field
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_name,
                docs=[{
                    "Title": "Horse rider",
                    "text_field": "hello",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "1"
                }],
                device="cpu",
                text_chunk_prefix="PREFIX: ",
                mappings={
                    "multimodal_fields_0": {
                        "type": "multimodal_combination",
                        "weights": {"text_field": 0.5,
                                    "image_field": 0.3}
                    }},
                tensor_fields=["multimodal_fields_0"]
            )
        )

        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_name,
                docs=[{
                    "Title": "Horse rider",
                    "text_field": "PREFIX: hello",
                    "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                    "_id": "2"
                }],
                device="cpu",
                mappings={
                    "multimodal_fields_0": {
                        "type": "multimodal_combination",
                        "weights": {"text_field": 0.5,
                                    "image_field": 0.3}
                    }},
                tensor_fields=["multimodal_fields_0"]
            )
        )

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.index_name, document_ids=["1", "2"],
            show_vectors=True
        )
        # Assert that the text field remains the same stored
        self.assertEqual(res["results"][0]["text_field"], "hello")
        # Assert that the text field embedding is equivalent to the embedding with the prefix
        self.assertTrue(np.allclose(res["results"][0]["_tensor_facets"][0]["_embedding"],
                                    res["results"][1]["_tensor_facets"][0]["_embedding"]))

    @mock.patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise)
    def test_add_prefix_to_queries(self, mock_vectorise):
        """Ensures that prefix gets added to each query."""
        # Single text query (prefix added)
        queries = [BulkSearchQueryEntity(q="hello", text_query_prefix="PREFIX: ", index=self.marqo_index)]
        prefixed_queries = tensor_search.add_prefix_to_queries(queries)
        self.assertEqual(prefixed_queries[0].q, "PREFIX: hello")

        # Dict query (text has prefix, image does not)
        queries = [BulkSearchQueryEntity(
            q={"text query": 0.5, "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5},
            text_query_prefix="PREFIX: ",
            index=self.marqo_index
        )]

        prefixed_queries = tensor_search.add_prefix_to_queries(queries)
        self.assertEqual(prefixed_queries[0].q, {"PREFIX: text query": 0.5,
                                                 "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5})


    @unittest.skip
    @patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise)
    def test_prefix_vectorise(self, mock_vectorise):
        """Ensures that vectorise is called on text with prefix"""

        # A) Add normal text document
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, tensor_fields=["text"]
        ))

        # B) Add same text document but WITH PREFIX (1 chunk)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, text_chunk_prefix="PREFIX: ", tensor_fields=["text"]
        ))

        # Vectorise should be called twice, once with no prefix and once with prefix
        self.assertEqual(len(mock_vectorise.call_args_list), 2)
        args, kwargs = mock_vectorise.call_args_list[0]
        self.assertEqual(kwargs["content"], ["hello"])
        args, kwargs = mock_vectorise.call_args_list[1]
        self.assertEqual(kwargs["content"], ["PREFIX: hello"])

    @unittest.skip
    def test_prefix_text_search(self, mock_vectorise):
        """Ensures that search query has prefix added to it for vectorisation."""

        # Add doc with no prefix
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, tensor_fields=["text"]
        ))
        # Add doc with prefix
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device, text_chunk_prefix="PREFIX: ", tensor_fields=["text"]
        ))

        tensor_search.search(config=self.config, index_name=self.index_name, text="searching",
                             device=self.config.default_device,
                             text_query_prefix="PREFIX: ")

        # Vectorise should be called 3 times - twice for add_documents, once for search
        self.assertEqual(len(mock_vectorise.call_args_list), 3)
        args, kwargs = mock_vectorise.call_args_list[-1]
        self.assertEqual(kwargs["content"], ["PREFIX: searching"])
