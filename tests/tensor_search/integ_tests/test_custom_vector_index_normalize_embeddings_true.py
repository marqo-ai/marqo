import os
from unittest import mock

import numpy as np

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.exceptions import ZeroMagnitudeVectorError
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.search import SearchContext
from marqo.vespa.models import VespaDocument, FeedBatchDocumentResponse, FeedBatchResponse
from tests.marqo_test import MarqoTestCase


class TestCustomVectorFieldWithIndexNormalizeEmbeddingsTrue(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Custom settings indexes
        test_unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True,
            normalize_embeddings=True, #Set normalize_embeddings to True
            distance_metric=DistanceMetric.Angular
        )


        test_structured_index_request = cls.structured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            normalize_embeddings=True,
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
            test_unstructured_index_request,
            test_structured_index_request,
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
        self.normalized_random_vector_1_nd_array = np.array(self.random_vector_1)
        self.normalized_random_vector_1 = (self.normalized_random_vector_1_nd_array / np.linalg.norm(np.array(self.random_vector_1), axis = -1, keepdims=True)).tolist()
        self.random_vector_2 = [i*2 for i in range(512)]
        self.normalized_random_vector_2 = (np.array(self.random_vector_2) / np.linalg.norm(np.array(self.random_vector_2), axis = -1, keepdims=True)).tolist()
        self.random_vector_3 = [1 / (i + 1) for i in range(512)]
        self.zero_vector = [0 for _ in range(512)]

        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_add_documents_with_custom_vector_normalize_embeddings_true(self):
        """
        This test method verifies that when documents with custom vector fields are added to both structured and
        unstructured indexes, the custom vectors are normalized as expected. Specifically this is checked for indexes
        where normalize_embeddings = True is set during index creation

        The test covers the following scenarios:
        1. Adding documents with custom vector fields to both structured and unstructured indexes.
        2. Ensuring that the custom vectors are normalized correctly.
        3. Verifying the response when a zero vector is provided.

        The test uses the `mock` library to mock the `feed_batch` method of the `VespaClient`.
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
                self.assertEqual(vespa_fields["marqo__short_string_fields"], {"my_custom_vector": "custom content is here!!"})
                self.assertEqual(vespa_fields["marqo__chunks"], ['my_custom_vector::custom content is here!!'])
                self.assertEqual(vespa_fields["marqo__embeddings"], {"0": self.normalized_random_vector_1})

            elif isinstance(index, StructuredMarqoIndex):
                self.assertEqual(vespa_fields["marqo__chunks_my_custom_vector"], ["custom content is here!!"])
                self.assertEqual(vespa_fields["marqo__embeddings_my_custom_vector"], {"0": self.normalized_random_vector_1})

            self.assertEqual(vespa_fields["marqo__vector_count"], 1)

    def test_add_documents_with_custom_vector_zero_vector_normalize_embeddings_true(self):
        """
        Test adding a document with a custom vector field where the vector is a zero vector,
        in an index where normalize_embeddings is set to True at the time of index creation.

        This test method verifies that when a document with a zero vector is added to both structured and
        unstructured indexes, the addition fails with an appropriate error message.

        The test covers the following scenarios:
        1. Adding a document with a zero vector to both structured and unstructured indexes.
        2. Ensuring that the addition fails with a "Zero magnitude vector detected, cannot normalize." error.
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

            add_documents_response = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[{
                            "_id": "0",
                            "my_custom_vector": {
                                "content": "custom content is here!!",
                                "vector": self.zero_vector
                            }
                        }],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

            if isinstance(index, UnstructuredMarqoIndex):
                self.assertEquals(add_documents_response.errors, True)
                self.assertEqual(add_documents_response.items[0].message, "Zero magnitude vector detected, cannot "
                                                                          "normalize.")
                self.assertEqual(add_documents_response.items[0].status, 400)
                self.assertEqual(add_documents_response.items[0].code, 'invalid_argument')
            elif isinstance(index, StructuredMarqoIndex):
                self.assertEquals(add_documents_response.errors, True)
                self.assertEqual(add_documents_response.items[0].message,
                                 "Zero magnitude vector detected, cannot normalize.")
                self.assertEqual(add_documents_response.items[0].status, 400)
                self.assertEqual(add_documents_response.items[0].code, 'invalid_argument')

    def test_search_with_custom_vector_field_normalize_embeddings_true(self):
        """
        Tensor search for the document, with highlights.

        This test method adds documents with custom vector fields to the index and performs tensor searches.
        It verifies that the search results and highlights are as expected when `normalize_embeddings` is set to True
        during index creation.

        The test covers the following scenarios:
        1. Adding documents with custom vector fields to both structured and unstructured indexes.
        2. Performing tensor searches with context matching custom vectors.
        3. Ensuring tensor search works even if the content is empty.
        4. Verifying that searching with normal text returns the expected text.

        The test uses the `self.subTest` context manager to create subtests for different index types.
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


    def test_search_with_custom_zero_vector_field_normalize_embeddings_true_wit(self):
        """
        Test tensor search with a custom vector field where the query vector is a zero vector,
        in an index where normalize_embeddings is set to True during index creation.

        This test method verifies that when documents with custom vector fields are added to the index and tensor searches
        are performed, an appropriate error is thrown during search.

        The test covers the following scenarios:
        1. Adding documents with custom vector fields to both structured and unstructured indexes.
        2. Performing tensor searches with zero vector as the query vector.
        3. Ensuring that the search fails with a ZeroMagnitudeVectorError.

        The test uses the `self.subTest` context manager to create subtests for different index types.
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
                            }
                        ],
                        device="cpu",
                        mappings=self.mappings if isinstance(index, UnstructuredMarqoIndex) else None,
                        tensor_fields=["my_custom_vector", "text_field"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Searching with context matching custom vector returns custom vector
                with self.assertRaises(ZeroMagnitudeVectorError) as e:
                    tensor_search.search(
                        config=self.config, index_name=index.name, text={"dummy text": 0},
                        search_method=enums.SearchMethod.TENSOR,
                        context=SearchContext(**{"tensor": [{"vector": self.zero_vector, "weight": 1}], })
                    )

                self.assertIn("Zero magnitude vector detected, cannot normalize.", str(e.exception))