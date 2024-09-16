import os
import unittest
import time

import numpy as np
from unittest import mock
from unittest.mock import Mock, patch
from marqo.core.embed.embed import Embed
from marqo.core.utils.prefix import determine_text_prefix, DeterminePrefixContentType
from marqo.api.models.embed_request import EmbedRequest
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.api import embed
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity
from marqo.core.models.marqo_index import StructuredMarqoIndex, FieldFeature, FieldType, Model
from marqo.core.models.marqo_index import FieldType, UnstructuredMarqoIndex, TextPreProcessing, \
    ImagePreProcessing, VideoPreProcessing, AudioPreProcessing, Model, DistanceMetric, VectorNumericType, \
    HnswConfig, TextSplitMethod, IndexType
from marqo.core.models.marqo_index_request import (StructuredMarqoIndexRequest, UnstructuredMarqoIndexRequest,
                                                   FieldRequest, MarqoIndexRequest)

from marqo.s2_inference import s2_inference
from tests.marqo_test import MarqoTestCase, TestImageUrls


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
        unstructured_index_e5 = cls.unstructured_marqo_index_request(
            model=Model(name='hf/e5-small'),
            treat_urls_and_pointers_as_images=True
        )
        unstructured_index_multimodal = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_index_with_model_default = cls.unstructured_marqo_index_request(
            model=Model(name="test_prefix"),
            treat_urls_and_pointers_as_images=False,
        )

        unstructured_index_with_override = cls.unstructured_marqo_index_request(
            model=Model(
                name="test_prefix",
                text_chunk_prefix="index-override: ",
                text_query_prefix="index-override: "
            ),
            treat_urls_and_pointers_as_images=True,
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
            unstructured_index_e5,
            unstructured_index_multimodal,
            unstructured_index_with_model_default,
            unstructured_index_with_override,
            structured_text_index,
            structured_multimodal_index,
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_index_1 = cls.indexes[0]
        cls.unstructured_index_e5 = cls.indexes[1]
        cls.unstructured_index_multimodal = cls.indexes[2]
        cls.unstructured_index_with_model_default = cls.indexes[3]
        cls.unstructured_index_with_override = cls.indexes[4]
        cls.structured_text_index = cls.indexes[5]
        cls.structured_multimodal_index = cls.indexes[6]



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
                ).dict(exclude_none=True, by_alias=True)["results"]
                retrieved_doc_a = res[0]
                retrieved_doc_b = res[1]
                retrieved_doc_c = res[2]

                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbedRequest(
                        content=["hello"],
                        content_type=None
                    ),
                    device="cpu"
                )

                # Chunk content: For A) and B), should be exactly the same. C) is different.
                self.assertEqual(retrieved_doc_a["text"], "hello")
                self.assertEqual(retrieved_doc_b["text"], "hello")
                self.assertEqual(retrieved_doc_c["text"], "PREFIX: hello")

                # Chunk embedding: For B) and C), should be exactly the same. A) is different.
                self.assertTrue(np.allclose(retrieved_doc_b["_tensor_facets"][0]["_embedding"],
                                            retrieved_doc_c["_tensor_facets"][0]["_embedding"]))
                self.assertFalse(np.allclose(retrieved_doc_a["_tensor_facets"][0]["_embedding"],
                                             retrieved_doc_c["_tensor_facets"][0]["_embedding"]))
                
                # embedding in document_b should be the same as direct embedding with no prefix
                self.assertTrue(np.allclose(retrieved_doc_a["_tensor_facets"][0]["_embedding"],
                                            embed_res["embeddings"][0]))
                
    def test_prefix_text_chunks_e5(self):
        """Ensures that the default prefix and the request level prefix are applied correctly.
        for the e5-small model."""

        for index in [self.unstructured_index_e5]:
            with self.subTest(index=index.type):
                # A) prefix should default to "passage: " with the e5-small model
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
                    device=self.config.default_device,
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # B) manually set prefix at the request level
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_b", "text": "hello"}], auto_refresh=True,
                    device=self.config.default_device, text_chunk_prefix="passage: ",
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # C) Set no prefix 
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[{"_id": "doc_c", "text": "hello"}], auto_refresh=True,
                    device=self.config.default_device, text_chunk_prefix="custom_prefix: ",
                    tensor_fields=["text"] if isinstance(index, UnstructuredMarqoIndex) else None
                ))

                # Get all documents (with vectors)
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name, document_ids=["doc_a", "doc_b", "doc_c"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)["results"]
                retrieved_doc_a = res[0]
                retrieved_doc_b = res[1]
                retrieved_doc_c = res[2]

                embed_res_document_prefix = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbedRequest(
                        content=["hello"],
                        content_type="document"
                    ),
                    device="cpu"
                )

                embed_res_no_prefix = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbedRequest(
                        content=["custom_prefix: hello"],
                        content_type=None
                    ),
                    device="cpu"
                )

                # Assert that the embedding in document_a is the same as embed_res_document_prefix with the prefix
                self.assertTrue(np.allclose(embed_res_document_prefix["embeddings"][0], retrieved_doc_a["_tensor_facets"][0]["_embedding"]))
                
                # Assert that the embedding in document_b is the same as the embedding in document_a
                self.assertTrue(np.allclose(retrieved_doc_a["_tensor_facets"][0]["_embedding"], retrieved_doc_b["_tensor_facets"][0]["_embedding"]))
                
                # Assert that the embedding in document_c is the same as the embedding with no prefix
                self.assertTrue(np.allclose(embed_res_no_prefix["embeddings"][0], retrieved_doc_c["_tensor_facets"][0]["_embedding"]))
                

    def test_prefix_multimodal(self):
        """Ensures that vectorise is called on text list with prefixes, but image list without."""

        for index in [self.unstructured_index_multimodal]:
            with self.subTest(index=index.type):
                # Add a multimodal doc with a text and image field
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "Title": "Horse rider",
                            "text_field": "hello",
                            "image_field": TestImageUrls.IMAGE1.value,
                            "_id": "1"
                        }],
                        device="cpu",
                        text_chunk_prefix="passage: ",
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
                            "text_field": "passage: hello",
                            "image_field": TestImageUrls.IMAGE1.value,
                            "_id": "2"
                        },
                        {
                            "Title": "Horse rider",
                            "text_field": "passage: passage: hello",
                            "image_field": TestImageUrls.IMAGE1.value,
                            "_id": "3"
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
                    config=self.config, index_name=index.name, document_ids=["1", "2", "3"],
                    show_vectors=True
                ).dict(exclude_none=True, by_alias=True)
                
                # Assert that the text field remains the same stored
                self.assertEqual(res["results"][0]["text_field"], "hello")
                # Assert that the text field embedding is equivalent to the embedding with the prefix
                self.assertTrue(np.allclose(res["results"][0]["_tensor_facets"][0]["_embedding"],
                                            res["results"][1]["_tensor_facets"][0]["_embedding"]))
                
                # Assert that no double prefixing happens in passage 1, so the embeddings of passage 1 != passage 3
                self.assertFalse(np.allclose(res["results"][0]["_tensor_facets"][0]["_embedding"],
                                            res["results"][2]["_tensor_facets"][0]["_embedding"]))


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
                       TestImageUrls.HIPPO_REALISTIC.value: 0.5},
                    text_query_prefix="PREFIX: ",
                    index=index
                )]

                prefixed_queries = tensor_search.add_prefix_to_queries(queries)
                self.assertEqual(prefixed_queries[0].q, {"PREFIX: text query": 0.5,
                                                         TestImageUrls.HIPPO_REALISTIC.value: 0.5})

    def test_determine_text_chunk_prefix(self):
        """
        Ensures proper priority order is followed when determining the chunk prefix.
        add docs request-level > index override-level > model default level
        """

        with self.subTest("All prefixes on (request level chosen)"):
            result = self.unstructured_index_with_override.model.get_text_chunk_prefix("request-level")
            self.assertEqual(result, "request-level")

        with self.subTest("Request and model default on (request level chosen)"):
            result = self.unstructured_index_with_model_default.model.get_text_chunk_prefix("request-level")
            self.assertEqual(result, "request-level")

        with self.subTest("Index override and model default on (index override chosen)"):
            result = self.unstructured_index_with_override.model.get_text_chunk_prefix(None)
            self.assertEqual(result, "index-override: ")

        with self.subTest("Only model default on (model default chosen)"):
            result = self.unstructured_index_with_model_default.model.get_text_chunk_prefix(None)
            self.assertEqual(result, "test passage: ")

        # doc_a should default to the override prefix
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.unstructured_index_with_override.name, docs=[{"_id": "doc_a", "text": "hello"}], auto_refresh=True,
            device=self.config.default_device,
            tensor_fields=["text"] if isinstance(self.unstructured_index_with_override, UnstructuredMarqoIndex) else None
        ))

        # Get all documents (with vectors)
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=self.unstructured_index_with_override.name, document_ids=["doc_a"],
            show_vectors=True
        ).dict(exclude_none=True, by_alias=True)

        # we hardcode the prefix into the text chunk and embed
        embed_res = embed(
            marqo_config=self.config, index_name=self.unstructured_index_with_override.name,
            embedding_request=EmbedRequest(
                content=["index-override: hello"],
                content_type=None
            ),
            device="cpu"
        )

        # We assert that the embeddings are equal
        with self.subTest("Embeddings are equal between overriden doc and direct embed"):
            self.assertTrue(np.allclose(embed_res["embeddings"][0], res["results"][0]["_tensor_facets"][0]["_embedding"]))


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
                search_query_embedding = vespa_query_kwargs["query_features"]["marqo__query_embedding"]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbedRequest(
                        content=["PREFIX: testing query"],
                        content_type=None
                    ),
                    device="cpu"
                )

                # Sanity check
                self.assertEqual(embed_res["content"], ["PREFIX: testing query"])

                # Assert vectors are equal. That is, the explicitly embedded query is the same as the query we sent
                # with set custom prefix
                self.assertTrue(np.allclose(embed_res["embeddings"][0], search_query_embedding))

    def test_backward_compatibility_no_prefix(self):
        """
        Ensures backward compatibility with older versions of Marqo that don't have prefix functionality.
        """
        
        mock_old_marqo_index = UnstructuredMarqoIndex(
            name="old_index",
            schema_name="old_index",
            type=IndexType.Unstructured,
            model=Model(
                name="hf/e5-small",
                text_chunk_prefix=None,
                text_query_prefix=None
            ),
            normalize_embeddings=True,
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=1000,
            text_preprocessing=TextPreProcessing(
                splitLength=6,
                splitOverlap=1,
                splitMethod=TextSplitMethod.Character
            ),
            image_preprocessing=ImagePreProcessing(),
            video_preprocessing=VideoPreProcessing(
                splitLength=20,
                splitOverlap=1,
            ),
            audio_preprocessing=AudioPreProcessing(
                splitLength=20,
                splitOverlap=1,
            ),
            distance_metric=DistanceMetric.DotProduct,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(
                ef_construction=100,
                m=42
            ),
            marqo_version="0.0.1",
            created_at=1,
            updated_at=1,
        )

        # Assert that when we attempt to get the text chunk prefix and text query prefix, it returns an empty string
        self.assertEqual(mock_old_marqo_index.model.get_text_chunk_prefix(), "")
        self.assertEqual(mock_old_marqo_index.model.get_text_query_prefix(), "")

    # NOTE: For tests on the prefix functionality on the embed endpoint, see tests_embed.py under 
    # integration tests.