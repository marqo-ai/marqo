from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search import enums
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity, ScoreModifier
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.models.search import SearchContext
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
import os
import pprint
import unittest
import httpx
import uuid


class TestCustomVectorField(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Custom settings indexes
        unstructured_custom_index = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True,
            normalize_embeddings=False,
            distance_metric=DistanceMetric.Angular
        )

        structured_custom_index = cls.structured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            normalize_embeddings=False,
            distance_metric=DistanceMetric.Angular,
            fields=[
                FieldRequest(
                    name="my_custom_vector",
                    type="custom_vector",
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(
                    name="text_field",
                    type="text",
                    features=[FieldFeature.LexicalSearch]),

                # For score modifiers test
                FieldRequest(
                    name="multiply",
                    type="float",
                    features=[FieldFeature.ScoreModifier]),

                # For searchable_attributes and filter tests
                FieldRequest(
                    name="my_custom_vector_2",
                    type="custom_vector",
                    features=[FieldFeature.Filter]),
                FieldRequest(
                    name="my_custom_vector_3",
                    type="custom_vector",
                    features=[FieldFeature.Filter]),

                # For lexical + searchable_attributes test
                FieldRequest(
                    name="exact_field",
                    type="text",
                    features=[FieldFeature.LexicalSearch]),
                FieldRequest(
                    name="barely_field",
                    type="text",
                    features=[FieldFeature.LexicalSearch]),

                # For multimodal mixed field tests
                FieldRequest(
                    name="multimodal_text",
                    type="text"),
                FieldRequest(
                    name="multimodal_image",
                    type="image_pointer"),
                FieldRequest(
                    name="my_multimodal",
                    type="multimodal_combination",
                    dependent_fields={"multimodal_text": 0.4, "multimodal_image": 0.6})
            ],
            tensor_fields=["my_custom_vector", "my_custom_vector_2", "my_custom_vector_3",
                           "text_field", "my_multimodal"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_custom_index,
            structured_custom_index
        ])

        cls.unstructured_custom_index = cls.indexes[0]
        cls.structured_custom_index = cls.indexes[1]

    def setUp(self):
        super().setUp()
        self.mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            }
        }

        # Using arbitrary values so they're easy to eyeball
        self.random_vector_1 = [1. for _ in range(512)]
        self.random_vector_2 = [i*2 for i in range(512)]
        self.random_vector_3 = [1 / (i + 1) for i in range(512)]

        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_add_documents_with_custom_vector_field(self):
        """
        Add a document with a custom vector field:
        mock call to vespa client
        Result will be slightly different for unstructured vs structured indexes.
        """

        for index in self.indexes:
            mock_feed_batch = mock.MagicMock()
            mock_feed_batch.return_value = FeedBatchResponse(
                responses=[FeedBatchDocumentResponse(
                    status=200,
                    pathId='/document/v1/aa5ed6d56e6aa4a048d95b496b79659f9/aa5ed6d56e6aa4a048d95b496b79659f9/docid/0',
                    id='id:aa5ed6d56e6aa4a048d95b496b79659f9:aa5ed6d56e6aa4a048d95b496b79659f9::0', message=None)
                ],
                errors=False)

            @mock.patch("marqo.vespa.vespa_client.VespaClient.feed_batch", mock_feed_batch)
            def run():
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "custom content is here!!",
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                return True

            assert run()

            call_args = mock_feed_batch.call_args_list
            assert len(call_args) == 1

            feed_batch_args = call_args[0].args
            self.assertIsInstance(feed_batch_args[0][0], VespaDocument)
            vespa_fields = feed_batch_args[0][0].fields

            if isinstance(index, UnstructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__strings"], ["custom content is here!!"])
                self.assertEqual(vespa_fields["marqo__long_string_fields"], {"my_custom_vector": "custom content is here!!"})
                self.assertEqual(vespa_fields["marqo__chunks"], ['my_custom_vector::custom content is here!!'])
                self.assertEqual(vespa_fields["marqo__embeddings"], {"0": self.random_vector_1})

            elif isinstance(index, StructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__chunks_my_custom_vector"], ["custom content is here!!"])
                self.assertEqual(vespa_fields["marqo__embeddings_my_custom_vector"], {"0": self.random_vector_1})

            self.assertEqual(vespa_fields["marqo__vector_count"], 1)

    def test_add_documents_with_custom_vector_field_no_content(self):
        """
        Add a document with a custom vector field with no content:
        Content should be autofilled with ""
        mock call to vespa client
        Structured and unstructured should have different calls.
        """
        for index in self.indexes:
            mock_feed_batch = mock.MagicMock()
            mock_feed_batch.return_value = FeedBatchResponse(
                responses=[FeedBatchDocumentResponse(
                    status=200,
                    pathId='/document/v1/aa5ed6d56e6aa4a048d95b496b79659f9/aa5ed6d56e6aa4a048d95b496b79659f9/docid/0',
                    id='id:aa5ed6d56e6aa4a048d95b496b79659f9:aa5ed6d56e6aa4a048d95b496b79659f9::0', message=None)
                ],
                errors=False)

            @mock.patch("marqo.vespa.vespa_client.VespaClient.feed_batch", mock_feed_batch)
            def run():
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                # No custom content
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                return True

            assert run()

            call_args = mock_feed_batch.call_args_list
            assert len(call_args) == 1

            feed_batch_args = call_args[0].args
            self.assertIsInstance(feed_batch_args[0][0], VespaDocument)
            vespa_fields = feed_batch_args[0][0].fields

            if isinstance(index, UnstructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__strings"], [""])
                self.assertEqual(vespa_fields["marqo__short_string_fields"],
                                 {"my_custom_vector": ""})
                self.assertEqual(vespa_fields["marqo__chunks"], ['my_custom_vector::'])
                self.assertEqual(vespa_fields["marqo__embeddings"], {"0": self.random_vector_1})

            elif isinstance(index, StructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__chunks_my_custom_vector"], [""])
                self.assertEqual(vespa_fields["marqo__embeddings_my_custom_vector"], {"0": self.random_vector_1})

            self.assertEqual(vespa_fields["marqo__vector_count"], 1)

    def test_add_documents_with_different_field_types(self):
        """
        Makes sure custom vector field doesn't mess up other kinds of fields
        Add a document with a custom vector field, multimodal, and standard:
        Mock vespa client call
        """
        # Mixed mapping to test both multimodal and custom vector
        multimodal_mappings = {
            "my_multimodal": {
                "type": "multimodal_combination",
                "weights": {
                    "multimodal_text": 0.4,
                    "multimodal_image": 0.6
                }
            },
            "my_custom_vector": {
                "type": "custom_vector"
            }
        }

        for index in self.indexes:
            mock_feed_batch = mock.MagicMock()
            mock_feed_batch.return_value = FeedBatchResponse(
                responses=[FeedBatchDocumentResponse(
                    status=200,
                    pathId='/document/v1/aa5ed6d56e6aa4a048d95b496b79659f9/aa5ed6d56e6aa4a048d95b496b79659f9/docid/0',
                    id='id:aa5ed6d56e6aa4a048d95b496b79659f9:aa5ed6d56e6aa4a048d95b496b79659f9::0', message=None)
                ],
                errors=False)

            @mock.patch("marqo.vespa.vespa_client.VespaClient.feed_batch", mock_feed_batch)
            def run():
                add_docs_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "multimodal_text": "blah",
                            "multimodal_image": 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png',
                            "my_custom_vector": {
                                "content": "custom content is here!!",
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=multimodal_mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "my_multimodal"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                return True

            assert run()

            call_args = mock_feed_batch.call_args_list
            assert len(call_args) == 1
            self.maxDiff = None
            feed_batch_args = call_args[0].args
            self.assertIsInstance(feed_batch_args[0][0], VespaDocument)
            vespa_fields = feed_batch_args[0][0].fields

            if isinstance(index, UnstructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__strings"],
                                 ['blah',
                                  'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png',
                                  'custom content is here!!'])
                self.assertEqual(vespa_fields["marqo__long_string_fields"],
                                 {'multimodal_image': 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png',
                                  'my_custom_vector': 'custom content is here!!'})
                self.assertEqual(vespa_fields["marqo__short_string_fields"],
                                 {'multimodal_text': 'blah'})
                self.assertEqual(vespa_fields["marqo__chunks"], ['my_custom_vector::custom content is here!!',
                                                                 'my_multimodal::{"multimodal_text": "blah", "multimodal_image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}'])
                self.assertEqual(vespa_fields["marqo__embeddings"]["0"], self.random_vector_1), # First vector is custom vector
                self.assertIn("1", vespa_fields["marqo__embeddings"])   # Just checking that multimodal vector is in embeddings, but not actually checking its value

            elif isinstance(index, StructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__chunks_my_custom_vector"], ['custom content is here!!'])
                self.assertEqual(vespa_fields["marqo__embeddings_my_custom_vector"], {"0": self.random_vector_1})
                self.assertEqual(vespa_fields["marqo__chunks_my_multimodal"], ['{"multimodal_text": "blah", "multimodal_image": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png"}'])
                self.assertIn("0", vespa_fields["marqo__embeddings_my_multimodal"])  # Just checking that multimodal vector is in embeddings, but not actually checking its value
            self.assertEqual(vespa_fields["marqo__vector_count"], 2)

    def test_add_documents_use_existing_tensors_with_custom_vector_field(self):
        """
        Add a document with a custom vector field and use existing tensors:
        Will not actually use existing tensors, as custom vector pipeline
        doesn't chunk or vectorise anyway.
        """

        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                # If we change the custom vector, doc should change
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "1 - custom content is here!!",
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                get_doc_1 = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)

                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "2 - custom content is here!!",
                                "vector": self.random_vector_2
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        use_existing_tensors=True
                    )
                )

                get_doc_2 = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)
                assert get_doc_1["my_custom_vector"] == "1 - custom content is here!!"
                assert get_doc_1[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_1
                assert get_doc_2["my_custom_vector"] == "2 - custom content is here!!"
                assert get_doc_2[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_2

                # If we do not, it should remain the same, no errors
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "2 - custom content is here!!",
                                "vector": self.random_vector_2
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        use_existing_tensors=True
                    )
                )

                get_doc_3 = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)
                assert get_doc_2["my_custom_vector"] == "2 - custom content is here!!"
                assert get_doc_2[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_2

    def test_get_document_with_custom_vector_field(self):
        """
        Add a document with a custom vector field:
        Get the doc, both fetched content and embedding must be correct
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "custom content is here!!",
                                "vector": self.random_vector_1
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Confirm get_document_by_id returns correct content
                res = tensor_search.get_document_by_id(
                    config=self.config, index_name=index.name,
                    document_id="0", show_vectors=True)

                # Check content is correct
                assert res["_id"] == "0"
                assert res["my_custom_vector"] == "custom content is here!!"

                # Check tensor facets and embedding are correct
                assert len(res[enums.TensorField.tensor_facets]) == 1
                assert res[enums.TensorField.tensor_facets][0]["my_custom_vector"] == "custom content is here!!"
                assert res[enums.TensorField.tensor_facets][0][enums.TensorField.embedding] == self.random_vector_1

    def test_get_documents_with_custom_vector_field(self):
        """
        Get multiple docs with custom vectors,
        both fetched content and embedding must be correct
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "0",
                                "my_custom_vector": {
                                    "content": "custom content is here!!",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "1",
                                "my_custom_vector": {
                                    "content": "second custom vector",
                                    "vector": self.random_vector_2
                                }
                            },
                            {
                                "_id": "2",
                                "my_custom_vector": {
                                    "content": "third custom vector",
                                    "vector": self.random_vector_3
                                }
                            },
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Confirm get_document_by_id returns correct content
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name,
                    document_ids=["0", "1", "2"], show_vectors=True)

                assert len(res["results"]) == 3

                # Check content is correct
                assert res["results"][0]["_id"] == "0"
                assert res["results"][0]["my_custom_vector"] == "custom content is here!!"
                # Check tensor facets and embedding are correct
                assert len(res["results"][0][enums.TensorField.tensor_facets]) == 1
                assert res["results"][0][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "custom content is here!!"
                assert np.allclose(res["results"][0][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_1)

                # Check content is correct
                assert res["results"][1]["_id"] == "1"
                assert res["results"][1]["my_custom_vector"] == "second custom vector"
                # Check tensor facets and embedding are correct
                assert len(res["results"][1][enums.TensorField.tensor_facets]) == 1
                assert res["results"][1][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "second custom vector"
                assert np.allclose(res["results"][1][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_2)

                # Check content is correct
                assert res["results"][2]["_id"] == "2"
                assert res["results"][2]["my_custom_vector"] == "third custom vector"
                # Check tensor facets and embedding are correct
                assert len(res["results"][2][enums.TensorField.tensor_facets]) == 1
                assert res["results"][2][enums.TensorField.tensor_facets][0]["my_custom_vector"] == "third custom vector"
                assert np.allclose(res["results"][2][enums.TensorField.tensor_facets][0][enums.TensorField.embedding], self.random_vector_3)

    def test_invalid_custom_vector_field_content(self):
        """
        Add a document with a custom vector field with invalid content/embedding/format
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                test_cases = [
                    # Wrong vector length
                    {"content": "custom content is here!!", "vector": [1.0, 1.0, 1.0]},
                    # Wrong content type
                    {"content": 12345, "vector": self.random_vector_1},
                    # Wrong vector type inside list (even if correct length)
                    {"content": "custom content is here!!", "vector": self.random_vector_1[:-1] + ["NOT A FLOAT"]},
                    # Field that shouldn't be there
                    {"content": "custom content is here!!", "vector": self.random_vector_1, "extra_field": "blah"},
                    # No vector
                    {"content": "custom content is here!!"},
                    # Nested dict inside custom vector content
                    {
                        "content": {
                            "content": "custom content is here!!",
                            "vector": self.random_vector_1
                        },
                        "vector": self.random_vector_1
                    },
                ]

                for case in test_cases:
                    res = tensor_search.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=[{
                                "_id": "0",
                                "my_custom_vector": case
                            }],
                            device="cpu",
                            mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                            tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                        )
                    )

                    assert res["errors"]
                    try:
                        tensor_search.get_document_by_id(config=self.config, index_name=index.name, document_id="0")
                        raise AssertionError
                    except DocumentNotFoundError:
                        pass

    def test_search_with_custom_vector_field(self):
        """
        Tensor search for the doc, with highlights
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom_vector_doc",
                                "my_custom_vector": {
                                    "content": "custom content is here!!",
                                    "vector": self.random_vector_1  # size is 512
                                }
                            },
                            {
                                "_id": "empty_content_custom_vector_doc",
                                "my_custom_vector": {
                                    "vector": self.random_vector_2  # size is 512
                                }
                            },
                            {
                                "_id": "normal_doc",
                                "text_field": "blah"
                            }
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "text_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Searching with context matching custom vector returns custom vector
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
                )

                assert res["hits"][0]["_id"] == "custom_vector_doc"
                assert res["hits"][0]["_score"] == 1.0
                assert res["hits"][0]["_highlights"][0]["my_custom_vector"] == "custom content is here!!"

                # Tensor search should work even if content is empty (highlight is empty string)
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_2, "weight": 1}], })
                )
                assert res["hits"][0]["_id"] == "empty_content_custom_vector_doc"
                assert res["hits"][0]["_score"] == 1.0
                assert res["hits"][0]["_highlights"][0]["my_custom_vector"] == ""

                # Searching with normal text returns text
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="blah",
                    search_method=enums.SearchMethod.TENSOR
                )
                assert res["hits"][0]["_id"] == "normal_doc"

    def test_lexical_search_with_custom_vector_field(self):
        """
        Lexical search for the doc
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom_vector_doc",
                                "my_custom_vector": {
                                    "content": "custom content is here!!",
                                    "vector": self.random_vector_1  # size is 512
                                }
                            },
                            {
                                "_id": "empty_content_custom_vector_doc",
                                "my_custom_vector": {
                                    "vector": self.random_vector_2  # size is 512
                                }
                            },
                            {
                                "_id": "normal_doc",
                                "text_field": "blah"
                            }
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "text_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Searching matching custom vector content returns custom vector
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="custom content is here!!",
                    search_method=enums.SearchMethod.LEXICAL
                )
                assert len(res["hits"]) == 1
                assert res["hits"][0]["_id"] == "custom_vector_doc"
                # Empty content doc should not be in lexical results
                for hit in res["hits"]:
                    assert hit["_id"] != "empty_content_custom_vector_doc"

                # Searching with normal text returns text
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="blah",
                    search_method=enums.SearchMethod.LEXICAL,
                )
                assert len(res["hits"]) == 1
                assert res["hits"][0]["_id"] == "normal_doc"
                # Empty content doc should not be in lexical results
                for hit in res["hits"]:
                    assert hit["_id"] != "empty_content_custom_vector_doc"


    def test_search_with_custom_vector_field_score_modifiers(self):
        """
        Search for the doc, with score modifiers
        """
        # custom vector cannot be used as score modifier, as it cannot be numeric.
        # Using another field as score modifier on a custom vector:
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                add_docs_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "doc0",
                                "my_custom_vector": {
                                    "content": "vec 1",
                                    "vector": self.random_vector_1  # size is 512
                                },
                                "multiply": 0.001  # Should make score tiny
                            },
                            {
                                "_id": "doc1",
                                "my_custom_vector": {
                                    "content": "vec 2",
                                    "vector": self.random_vector_2  # size is 512
                                },
                                "multiply": 1000.  # Should make score huge
                            },
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Normal search should favor doc0
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
                )
                assert res["hits"][0]["_id"] == "doc0"

                # Search with score modifiers multiply should favor doc1
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    score_modifiers=ScoreModifier(**{"multiply_score_by":
                                                         [{"field_name": "multiply",
                                                           "weight": 1}
                                                          ]
                                                     })
                )
                assert res["hits"][0]["_id"] == "doc1"

    def test_search_with_custom_vector_field_filter_string(self):
        """
        Search for the doc, with filter string
        """

        new_mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            },
            "my_custom_vector_2": {
                "type": "custom_vector"
            },
            "my_custom_vector_3": {
                "type": "custom_vector"
            },
        }

        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                add_docs_res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom vector doc 1",
                                "my_custom_vector": {
                                    "content": "red blue yellow",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "custom vector doc 2",
                                "my_custom_vector_2": {
                                    "content": "red",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "custom vector doc 3",
                                "my_custom_vector_2": {
                                    "content": "blue",
                                    "vector": self.random_vector_1
                                },
                                "my_custom_vector_3": {
                                    "content": "chocolate",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "custom vector doc 4",
                                "my_custom_vector": {
                                    "content": "yellow",
                                    "vector": self.random_vector_1
                                },
                                # Empty content field. Should not affect the document getting returned.
                                "my_custom_vector_2": {
                                    "vector": self.random_vector_1
                                },
                                "my_custom_vector_3": {
                                    "content": "chocolate",
                                    "vector": self.random_vector_1
                                }
                            },
                        ],
                        device="cpu",
                        mappings=new_mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "my_custom_vector_2", "my_custom_vector_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # No filter: all docs are returned.
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    result_count=10
                )
                res_ids = set([hit["_id"] for hit in res["hits"]])
                assert res_ids == {"custom vector doc 1", "custom vector doc 2", "custom vector doc 3", "custom vector doc 4"}

                # Filter: custom vector 3 has chocolate
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    filter="my_custom_vector_3:chocolate", result_count=10
                )
                res_ids = set([hit["_id"] for hit in res["hits"]])
                assert res_ids == {"custom vector doc 3", "custom vector doc 4"}

                # Filter: AND statement
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    filter="my_custom_vector_3:chocolate AND my_custom_vector_2:blue", result_count=10
                )
                res_ids = set([hit["_id"] for hit in res["hits"]])
                assert res_ids == {"custom vector doc 3"}

                # Filter: OR statement
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    filter="my_custom_vector:red OR my_custom_vector_2:red", result_count=10
                )
                res_ids = set([hit["_id"] for hit in res["hits"]])
                assert res_ids == {"custom vector doc 2"}

                # Filter: parenthesis
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    filter="my_custom_vector:(red blue yellow)", result_count=10
                )
                res_ids = set([hit["_id"] for hit in res["hits"]])
                assert res_ids == {"custom vector doc 1"}

    def test_search_with_custom_vector_field_searchable_attributes(self):
        """
        Searchable attributes are only available for structured indexes.
        """
        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                # Skip this test for unstructured indexes.
                if isinstance(index, UnstructuredMarqoIndex):
                    break

                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom vector doc 1",
                                "my_custom_vector": {
                                    "content": "doesn't matter",
                                    "vector": self.random_vector_1
                                }
                            },
                            {
                                "_id": "custom vector doc 2",
                                "my_custom_vector_2": {
                                    "content": "doesn't matter",
                                    "vector": self.random_vector_2
                                }
                            },
                            {
                                "_id": "custom vector doc 3",
                                "my_custom_vector_3": {
                                    "content": "doesn't matter",
                                    "vector": self.random_vector_3
                                }
                            },
                        ],
                        device="cpu"
                    )
                )

                # All searchable attributes
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    searchable_attributes=["my_custom_vector", "my_custom_vector_2", "my_custom_vector_3"]
                )
                assert res["hits"][0]["_id"] == "custom vector doc 1"

                # Only 2 and 3
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    searchable_attributes=["my_custom_vector_2", "my_custom_vector_3"]
                )
                assert res["hits"][0]["_id"] == "custom vector doc 2"

                # Only 3
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text={"dummy text": 0},
                    search_method=enums.SearchMethod.TENSOR,
                    context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
                    searchable_attributes=["my_custom_vector_3"]
                )
                assert res["hits"][0]["_id"] == "custom vector doc 3"

    def test_lexical_search_with_custom_vector_field_searchable_attributes(self):
        """
        Search for the doc with lexical search, with searchable attributes
        """

        for index in self.indexes:
            with self.subTest(f"Index: {index.name}, type: {index.type}"):
                # Skip this test for unstructured indexes.
                if isinstance(index, UnstructuredMarqoIndex):
                    break
                tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {
                                "_id": "custom vector doc",
                                "my_custom_vector": {
                                    "content": "toxt to search",  # almost matching
                                    "vector": self.random_vector_1  # size is 512
                                }
                            },
                            {
                                "_id": "barely matching doc",
                                "barely_field": "random words search"
                            },
                            {
                                "_id": "exactly matching doc",
                                "exact_field": "text to search"
                            }
                        ],
                        device="cpu"
                    )
                )

                # All searchable attributes
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="text to search",
                    search_method=enums.SearchMethod.LEXICAL,
                    searchable_attributes=["my_custom_vector", "barely_field", "exact_field"]
                )
                assert res["hits"][0]["_id"] == "exactly matching doc"

                # Only custom and barely matching
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="text to search",
                    search_method=enums.SearchMethod.LEXICAL,
                    searchable_attributes=["my_custom_vector", "barely_field"]
                )
                assert res["hits"][0]["_id"] == "custom vector doc"

                # Only barely matching
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="text to search",
                    search_method=enums.SearchMethod.LEXICAL,
                    searchable_attributes=["barely_field"]
                )
                assert res["hits"][0]["_id"] == "barely matching doc"

                # Only custom vector
                res = tensor_search.search(
                    config=self.config, index_name=index.name, text="text to search",
                    search_method=enums.SearchMethod.LEXICAL,
                    searchable_attributes=["my_custom_vector"]
                )
                assert res["hits"][0]["_id"] == "custom vector doc"

    def test_custom_vector_subfield_of_multimodal_should_fail_structured(self):
        """
        When attempting to create a structured index, a custom vector can not be a subfield of a multimodal field.
        Remove this test when this functionality becomes available.
        """

        with self.assertRaises(base_exceptions.InvalidArgumentError) as cm:
            bad_index_request = self.structured_marqo_index_request(
                fields=[
                    FieldRequest(
                        name="my_custom_vector",
                        type="custom_vector"),
                    FieldRequest(
                        name="bad_multimodal",
                        type="multimodal_combination",
                        dependent_fields={"my_custom_vector": 0.5})
                ],
                tensor_fields=["my_custom_vector", "bad_multimodal"]
            )

        self.assertIn("cannot be a subfield of a multimodal field", cm.exception.message)

    def test_custom_vector_subfield_of_multimodal_should_fail_unstructured(self):
        """
        When attempting to add documents to an unstructured index, a custom vector can not be a subfield of a multimodal field.
        Remove this test when this functionality becomes available.
        """

        add_docs_res = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.unstructured_custom_index.name,
                docs=[
                    {
                        "_id": "doc0",
                        "my_custom_vector": {
                            "content": "vec 1",
                            "vector": self.random_vector_1  # size is 512
                        },
                    }
                ],
                device="cpu",
                tensor_fields=["my_custom_vector", "bad_multimodal"],
                mappings={
                    "my_custom_vector": {
                        "type": "custom_vector"
                    },
                    "bad_multimodal": {
                        "type": "multimodal_combination",
                        "weights": {
                            "my_custom_vector": 0.5
                        }
                    }
                }
            )
        )

        self.assertEqual(add_docs_res["errors"], True)
        self.assertEqual(add_docs_res["items"][0]["code"], "invalid_argument")
        self.assertEqual(add_docs_res["items"][0]["status"], 400)
        self.assertIn("Multimodal subfields must be strings", add_docs_res["items"][0]["error"])


    @unittest.skip
    def test_search_with_custom_vector_field_boosting(self):
        """
        SKIPPED WHILE BOOSTING IS NOT YET IMPLEMENTED.
        Search for the doc, with boosting
        """
        mappings = {
            "my_custom_vector_1": {
                "type": "custom_vector"
            },
            "my_custom_vector_2": {
                "type": "custom_vector"
            },
        }

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                docs=[
                    {
                        "_id": "doc0",
                        "my_custom_vector_1": {
                            "content": "vec 1",
                            "vector": self.random_vector_1  # size is 512
                        },
                    },
                    {
                        "_id": "doc1",
                        "my_custom_vector_2": {
                            "content": "vec 2",
                            "vector": self.random_vector_2  # size is 512
                        },
                    },
                ],
                device="cpu", mappings=mappings
            )
        )

        # Normal search should favor doc0
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], })
        )
        assert res["hits"][0]["_id"] == "doc0"

        # Search with boosting should favor doc1
        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text={"dummy text": 0},
            search_method=enums.SearchMethod.TENSOR,
            context=SearchContext(**{"tensor": [{"vector": self.random_vector_1, "weight": 1}], }),
            boost={"my_custom_vector_2": [5, 1]}
        )
        assert res["hits"][0]["_id"] == "doc1"