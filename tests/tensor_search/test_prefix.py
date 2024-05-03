import os
import unittest

import numpy as np
from unittest import mock
from unittest.mock import Mock, patch
from marqo.core.embed.embed import Embed
from marqo.api.models.embed_request import EmbedRequest
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.api import embed
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity
from marqo.core.models.marqo_index import UnstructuredMarqoIndex, StructuredMarqoIndex, FieldFeature, FieldType, Model
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference import s2_inference
from tests.marqo_test import MarqoTestCase


def pass_through_vectorise(*args, **kwargs):
    return s2_inference.vectorise(*args, **kwargs)


class TestPrefix(MarqoTestCase):
    """
    Tests the prefix logic for adding prefixes to text fields and search queries.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes
        unstructured_index_1 = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
            treat_urls_and_pointers_as_images=True
        )
        unstructured_index_2 = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
            treat_urls_and_pointers_as_images=True
        )

        # STRUCTURED indexes
        structured_text_index = cls.structured_marqo_index_request(
            model=Model(name="random/small"),
            fields=[
                FieldRequest(name="text", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])
            ],
            tensor_fields=["text"]
        )

        structured_multimodal_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="TITLE", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="multimodal_fields", type=FieldType.MultimodalCombination,
                             dependent_fields={"text_field": 0.5, "image_field": 0.5})
            ],
            tensor_fields=["multimodal_fields"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_index_1,
            unstructured_index_2,
            structured_text_index,
            structured_multimodal_index
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_index_1 = cls.indexes[0]
        cls.unstructured_index_2 = cls.indexes[1]
        cls.structured_text_index = cls.indexes[2]
        cls.structured_multimodal_index = cls.indexes[3]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_prefix_text_chunks(self):
        """Ensures that when adding documents with a prefix, each chunk has the prefix included in the vector,
        but the actual chunk text does not have the prefix."""

        for index in [self.unstructured_index_1, self.structured_text_index]:
            with self.subTest(index=index.type):
                # A) Add normal text document (1 chunk)
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
                    device=self.config.default_device,
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # B) Add same text document but WITH PREFIX (1 chunk)
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
                    device=self.config.default_device, text_chunk_prefix="PREFIX: ",
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # C) Add document with prefix built into text itself (1 chunk)
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_c", "text": "PREFIX: hello"}], auto_refresh=True,
                    device=self.config.default_device,
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # Get all documents (with vectors)
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name, document_ids=["doc_a", "doc_b", "doc_c"],
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

        for index in [self.unstructured_index_1]:
            with self.subTest(index=index.type):
                # Add a multimodal doc with a text and image field
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "Title": "Horse rider",
                            "text_field": "hello",
                            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "_id": "1"
                        }],
                        device="cpu",
                        text_chunk_prefix="PREFIX: ",
                        mappings={
                            "multimodal_fields": {
                                "type": "multimodal_combination",
                                "weights": {"text_field": 0.5,
                                            "image_field": 0.3}
                            }} if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["multimodal_fields"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "Title": "Horse rider",
                            "text_field": "PREFIX: hello",
                            "image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
                            "_id": "2"
                        }],
                        device="cpu",
                        mappings={
                            "multimodal_fields": {
                                "type": "multimodal_combination",
                                "weights": {"text_field": 0.5,
                                            "image_field": 0.3}
                            }} if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["multimodal_fields"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Get all documents (with vectors)
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name, document_ids=["1", "2"],
                    show_vectors=True
                )
                print(res)
                # Assert that the text field remains the same stored
                self.assertEqual(res["results"][0]["text_field"], "hello")
                # Assert that the text field embedding is equivalent to the embedding with the prefix
                self.assertTrue(np.allclose(res["results"][0]["_tensor_facets"][0]["_embedding"],
                                            res["results"][1]["_tensor_facets"][0]["_embedding"]))

    @mock.patch("marqo.s2_inference.s2_inference.vectorise", side_effect=pass_through_vectorise)
    def test_add_prefix_to_multimodal_queries(self, mock_vectorise):
        """Ensures that prefix gets added to each query."""
        for index in [self.unstructured_index_1, self.structured_text_index]:
            with self.subTest(index=index.type):
                # Single text query (prefix added)
                queries = [BulkSearchQueryEntity(q="hello", text_query_prefix="PREFIX: ", index=index)]
                prefixed_queries = tensor_search.add_prefix_to_queries(queries)
                self.assertEqual(prefixed_queries[0].q, "PREFIX: hello")

                # Dict query (text has prefix, image does not)
                queries = [BulkSearchQueryEntity(
                    q={"text query": 0.5,
                       "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5},
                    text_query_prefix="PREFIX: ",
                    index=index
                )]

                prefixed_queries = tensor_search.add_prefix_to_queries(queries)
                self.assertEqual(prefixed_queries[0].q, {"PREFIX: text query": 0.5,
                                                         "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png": 0.5})

    def test_prefix_text_search(self):
        """Ensures that search query has prefix added to it for vectorisation."""
        for index in [self.unstructured_index_1, self.structured_text_index]:
            with self.subTest(index=index.type):
                original_query = self.config.vespa_client.query

                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                mock_vespa_client_query = unittest.mock.MagicMock()
                mock_vespa_client_query.side_effect = pass_through_query

                @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                def run():
                    tensor_search.search(
                        config=self.config, index_name=index.name, text="testing query",
                        search_method=enums.SearchMethod.TENSOR, text_query_prefix="PREFIX: "
                    )
                    return True

                self.assertTrue(run())

                call_args = mock_vespa_client_query.call_args_list
                self.assertEqual(len(call_args), 1)

                vespa_query_kwargs = call_args[0].kwargs
                if isinstance(index, UnstructuredMarqoIndex):
                    embedding_key = "embedding_query"
                elif isinstance(index, StructuredMarqoIndex):
                    embedding_key = "marqo__query_embedding"
                search_query_embedding = vespa_query_kwargs["query_features"][embedding_key]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbedRequest(
                        content=["PREFIX: testing query"],
                        content_type=""
                    ),
                    device="cpu"
                )

                # Sanity check
                self.assertEqual(embed_res["content"], ["PREFIX: testing query"])

                # Assert vectors are equal. That is, the explicitly embedded query is the same as the query we sent
                # with set custom prefix
                self.assertTrue(np.allclose(embed_res["embeddings"][0], search_query_embedding))
