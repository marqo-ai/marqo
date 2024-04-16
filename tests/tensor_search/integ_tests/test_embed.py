from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search import enums
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity, ScoreModifier
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search.api import embed
import numpy as np
import requests
import json
from unittest import mock
from unittest.mock import patch
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.api.exceptions import MarqoWebError, IndexNotFoundError, InvalidArgError, DocumentNotFoundError
import marqo.exceptions as base_exceptions
from marqo.core.models.marqo_index import *
from marqo.vespa.models import VespaDocument, QueryResult, FeedBatchDocumentResponse, FeedBatchResponse, \
    FeedDocumentResponse
from marqo.vespa.models.query_result import Root, Child, RootFields
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
from marqo.api.models.embed_request import EmbeddingRequest
import os
import pprint
import unittest
import httpx
import uuid


class TestEmbed(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Mock query result
        cls.mock_query_result = QueryResult(
            root = Root(id='toplevel', relevance=1.0, source=None, label=None, value=None,
                        errors=None, children=[
                    Child(id='index:content_default/0/674f3c2ce3ae7b71e3e805be', relevance=0.5975484726281337,
                          source='content_default', label=None, value=None, coverage=None, errors=None, children=None,
                          fields={'matchfeatures': {
                              'closest(marqo__embeddings)': {'type': 'tensor<float>(p{})', 'cells': {'1': 1.0}}},
                                  'sddocname': 'a06babea28d0f408ebb0e7be130f86f40', 'marqo__id': '5678',
                                  'marqo__strings': ['Exact match hehehe efgh '],
                                  'marqo__long_string_fields': {'abc': 'Exact match hehehe efgh '},
                                  'marqo__short_string_fields': {},
                                  'marqo__chunks': ['abc::Exact match hehehe efgh']})], fields=RootFields(total_count=1)),
        )

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        unstructured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True
        )

        unstructured_image_index_with_random_model = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
            treat_urls_and_pointers_as_images=True
        )

        # STRUCTURED indexes
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
            ],

            tensor_fields=["text_field_1", "text_field_2"]
        )

        structured_default_image_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "image_field_1", "image_field_2"]
        )

        structured_image_index_with_random_model = cls.structured_marqo_index_request(
            model=Model(name='random/small'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer)
            ],
            tensor_fields=["text_field_1", "text_field_2", "image_field_1"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            unstructured_default_image_index,
            unstructured_image_index_with_random_model,
            structured_default_text_index,
            structured_default_image_index,
            structured_image_index_with_random_model
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.unstructured_default_image_index = cls.indexes[1]
        cls.unstructured_image_index_with_random_model = cls.indexes[2]
        cls.structured_default_text_index = cls.indexes[3]
        cls.structured_default_image_index = cls.indexes[4]
        cls.structured_image_index_with_random_model = cls.indexes[5]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_embed_equivalent_to_add_docs(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                add_docs_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "0",
                                "text_field_1": "I am the GOAT."
                            }
                        ],
                        device="cpu",
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Get the added document embedding
                get_docs_res = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)

                self.assertEqual(get_docs_res["_id"], "0")
                self.assertEqual(len(get_docs_res[enums.TensorField.tensor_facets]), 1)

                get_docs_embedding = get_docs_res[enums.TensorField.tensor_facets][0][enums.TensorField.embedding]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbeddingRequest(
                        content=["I am the GOAT."]
                    ),
                    device="cpu"
                )

                # Assert vectors are equal
                self.assertEqual(embed_res["content"], ["I am the GOAT."])
                assert np.allclose(embed_res["embeddings"][0], get_docs_embedding)

    def test_embed_equivalent_to_search_text(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                original_query = self.config.vespa_client.query
                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                mock_vespa_client_query = unittest.mock.MagicMock()
                mock_vespa_client_query.side_effect = pass_through_query

                @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                def run():
                    tensor_search.search(
                        config=self.config, index_name=index.name, text="I am the GOAT.",
                        search_method=enums.SearchMethod.TENSOR
                    )
                    return True
                assert run()

                call_args = mock_vespa_client_query.call_args_list
                assert len(call_args) == 1

                vespa_query_kwargs = call_args[0].kwargs
                if isinstance(index, UnstructuredMarqoIndex):
                    embedding_key = "embedding_query"
                elif isinstance(index, StructuredMarqoIndex):
                    embedding_key = "marqo__query_embedding"
                search_query_embedding = vespa_query_kwargs["query_features"][embedding_key]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbeddingRequest(
                        content=["I am the GOAT."]
                    ),
                    device="cpu"
                )

                # Assert vectors are equal
                self.assertEqual(embed_res["content"], ["I am the GOAT."])
                assert np.allclose(embed_res["embeddings"][0], search_query_embedding)

    def test_embed_equivalent_to_search_image(self):
        for index in [self.unstructured_default_image_index, self.structured_default_image_index]:
            with self.subTest(index=index.type):
                image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg"
                original_query = self.config.vespa_client.query

                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                mock_vespa_client_query = unittest.mock.MagicMock()
                mock_vespa_client_query.side_effect = pass_through_query

                @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                def run():
                    tensor_search.search(
                        config=self.config, index_name=index.name, text=image_url,
                        search_method=enums.SearchMethod.TENSOR
                    )
                    return True

                assert run()

                call_args = mock_vespa_client_query.call_args_list
                assert len(call_args) == 1

                vespa_query_kwargs = call_args[0].kwargs
                if isinstance(index, UnstructuredMarqoIndex):
                    embedding_key = "embedding_query"
                elif isinstance(index, StructuredMarqoIndex):
                    embedding_key = "marqo__query_embedding"
                search_query_embedding = vespa_query_kwargs["query_features"][embedding_key]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbeddingRequest(
                        content=[image_url]
                    ),
                    device="cpu"
                )

                # Assert vectors are equal
                self.assertEqual(embed_res["content"], [image_url])
                assert np.allclose(embed_res["embeddings"][0], search_query_embedding)

    def test_embed_with_image_download_headers_and_model_auth(self):
        """
        Ensure that vectorise is called with the correct image_download_headers and model_auth
        when using the embed endpoint.
        """
        for index in [self.unstructured_default_image_index, self.structured_default_image_index]:
            with self.subTest(index=index.type):
                image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg"
                vectorise = s2_inference.vectorise
                def pass_through_vectorise(*arg, **kwargs):
                    """Vectorise will behave as usual, but we will be able to see the call list
                    via mock
                    Set image download headers and model auth to None so there's no error out.
                    """
                    kwargs["image_download_headers"] = None
                    kwargs["model_auth"] = None
                    return vectorise(*arg, **kwargs)

                mock_vectorise = unittest.mock.MagicMock()
                mock_vectorise.side_effect = pass_through_vectorise
                @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
                def run():
                    embed_res = embed(
                        marqo_config=self.config, index_name=index.name,
                        embedding_request=EmbeddingRequest(
                            content=[image_url],
                            image_download_headers={"Authorization": "my secret key"},
                            modelAuth=ModelAuth(s3=S3Auth(
                                aws_access_key_id='12345',
                                aws_secret_access_key='this-is-a-secret'))
                        ),
                        device="cpu"
                    )
                    return True

                assert run()

                call_args = mock_vectorise.call_args_list
                assert len(call_args) == 1

                vectorise_kwargs = call_args[0].kwargs
                self.assertEqual(vectorise_kwargs["image_download_headers"], {"Authorization": "my secret key"})
                self.assertEqual(vectorise_kwargs["model_auth"], ModelAuth(s3=S3Auth(
                                aws_access_key_id='12345',
                                aws_secret_access_key='this-is-a-secret')))

    def test_embed_equivalent_to_search_weighted_dict(self):
        """
        Ensure that a weighted dict embed request will return the same result as a weighted dict search query.
        We will mix both text and images here.
        """
        for index in [self.unstructured_default_image_index, self.structured_default_image_index]:
            with self.subTest(index=index.type):
                image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg"
                original_query = self.config.vespa_client.query
                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                mock_vespa_client_query = unittest.mock.MagicMock()
                mock_vespa_client_query.side_effect = pass_through_query

                @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                def run():
                    tensor_search.search(
                        config=self.config, index_name=index.name,
                        text={"I am the GOATest of all time.": 0.7, image_url: 0.3},
                        search_method=enums.SearchMethod.TENSOR
                    )
                    return True

                assert run()

                call_args = mock_vespa_client_query.call_args_list
                assert len(call_args) == 1

                vespa_query_kwargs = call_args[0].kwargs
                if isinstance(index, UnstructuredMarqoIndex):
                    embedding_key = "embedding_query"
                elif isinstance(index, StructuredMarqoIndex):
                    embedding_key = "marqo__query_embedding"
                search_query_embedding = vespa_query_kwargs["query_features"][embedding_key]

                # Embed request the same text
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbeddingRequest(
                        content=[{"I am the GOATest of all time.": 0.7, image_url: 0.3}]
                    ),
                    device="cpu"
                )

                # Assert vectors are equal
                self.assertEqual(embed_res["content"], [{"I am the GOATest of all time.": 0.7, image_url: 0.3}])
                assert np.allclose(embed_res["embeddings"][0], search_query_embedding)


    def test_embed_equivalent_to_search_multiple_content(self):
        for index in [self.unstructured_default_image_index, self.structured_default_image_index]:
            with self.subTest(index=index.type):
                image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg"
                original_query = self.config.vespa_client.query
                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                # Mixed list: text, text, image, dict
                sample_content_list = ["GOAT #1", "GOAT #2", image_url, {"The inner GOAT.": 0.65, image_url: 0.42}]

                # Collect embeddings from each search query
                search_query_embeddings = []
                for item in sample_content_list:
                    mock_vespa_client_query = unittest.mock.MagicMock()
                    mock_vespa_client_query.side_effect = pass_through_query

                    @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                    def run():
                        tensor_search.search(
                            config=self.config, index_name=index.name,
                            text=item,
                            search_method=enums.SearchMethod.TENSOR
                        )
                        return True

                    assert run()

                    call_args = mock_vespa_client_query.call_args_list
                    assert len(call_args) == 1

                    vespa_query_kwargs = call_args[0].kwargs
                    if isinstance(index, UnstructuredMarqoIndex):
                        embedding_key = "embedding_query"
                    elif isinstance(index, StructuredMarqoIndex):
                        embedding_key = "marqo__query_embedding"
                    search_query_embeddings.append(vespa_query_kwargs["query_features"][embedding_key])

                # Embed request the content list
                embed_res = embed(
                    marqo_config=self.config, index_name=index.name,
                    embedding_request=EmbeddingRequest(
                        content=sample_content_list
                    ),
                    device="cpu"
                )

                # Assert vectors are equal
                self.assertEqual(embed_res["content"], sample_content_list)
                for i in range(len(sample_content_list)):
                    assert np.allclose(embed_res["embeddings"][i], search_query_embeddings[i], atol=1e-6)

    def test_embed_empty_content_list(self):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                with self.assertRaises(ValidationError) as e:
                    embed_res = embed(
                        marqo_config=self.config, index_name=index.name,
                        embedding_request=EmbeddingRequest(
                            content=[]
                        ),
                        device="cpu"
                    )

                self.assertIn("should not be empty", str(e.exception))

    def test_embed_invalid_content_type(self):
        test_cases = [
            ({"key": "not a number"}, "not a valid float"),  # dict with wrong typed value
            ([{"key": "value"}], "not a valid float")  # list of dict with wrong typed value
        ]
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                for content, error_message in test_cases:
                    with self.subTest(content=content):
                        with self.assertRaises(ValidationError) as e:
                            embed(
                                marqo_config=self.config, index_name=index.name,
                                embedding_request=EmbeddingRequest(
                                    content=content
                                ),
                                device="cpu"
                            )
                    self.assertIn(error_message, str(e.exception))