import os
import uuid
from unittest import mock

import marqo.core.exceptions as core_exceptions
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.structured_vespa_index import common
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo import exceptions as base_exceptions
import unittest


class TestHybridSearch(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        unstructured_default_text_index_encoded_name = cls.unstructured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', '')
        )

        unstructured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True
        )

        unstructured_image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
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
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_4", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_5", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_6", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_7", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_8", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="int_field_1", type=FieldType.Int,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="float_field_1", type=FieldType.Float,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="bool_field_1", type=FieldType.Bool,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="bool_field_2", type=FieldType.Bool,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
                FieldRequest(name="long_field_1", type=FieldType.Long, features=[FieldFeature.Filter]),
                FieldRequest(name="double_field_1", type=FieldType.Double, features=[FieldFeature.Filter]),
                FieldRequest(name="custom_vector_field_1", type=FieldType.CustomVector, features=[FieldFeature.Filter]),
                FieldRequest(name="multimodal_field_1", type=FieldType.MultimodalCombination,
                             dependent_fields={"text_field_7": 0.1, "text_field_8": 0.1})
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3",
                           "text_field_4", "text_field_5", "text_field_6",
                           "custom_vector_field_1", "multimodal_field_1"]
        )

        structured_default_text_index_encoded_name = cls.structured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])
            ],

            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
        )

        structured_default_image_index = cls.structured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(name="list_field_1", type=FieldType.ArrayText,
                             features=[FieldFeature.Filter]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "text_field_3", "image_field_1", "image_field_2"]
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

        structured_text_index_single_field = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])
            ],
            tensor_fields=["text_field_1"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            unstructured_default_text_index_encoded_name,
            unstructured_default_image_index,
            unstructured_image_index_with_chunking,
            unstructured_image_index_with_random_model,
            structured_default_text_index,
            structured_default_text_index_encoded_name,
            structured_default_image_index,
            structured_image_index_with_random_model,
            structured_text_index_single_field
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.unstructured_default_text_index_encoded_name = cls.indexes[1]
        cls.unstructured_default_image_index = cls.indexes[2]
        cls.unstructured_image_index_with_chunking = cls.indexes[3]
        cls.unstructured_image_index_with_random_model = cls.indexes[4]
        cls.structured_default_text_index = cls.indexes[5]
        cls.structured_default_text_index_encoded_name = cls.indexes[6]
        cls.structured_default_image_index = cls.indexes[7]
        cls.structured_image_index_with_random_model = cls.indexes[8]
        cls.structured_text_index_single_field = cls.indexes[9]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_hybrid_search_calls_correct_vespa_query(self):
        """
        Test all hybrid search methods call the correct vespa queries.
        """

        # Retrieval and Ranking Method Mappings
        RETRIEVAL_METHOD_YQL_MAPPING = {
            # Use OR
            RetrievalMethod.Disjunction: '(({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}'
                                         'nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding))) '
                                         'OR weakAnd(default contains "dogs")',
            # Use rank(tensor, lexical)
            RetrievalMethod.Tensor: 'rank((({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}'
                                    'nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding))), '
                                    'weakAnd(default contains "dogs"))',
            # Use rank(lexical, tensor)
            RetrievalMethod.Lexical: 'rank(weakAnd(default contains "dogs"), '
                                     '(({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}'
                                    'nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding))))'
        }

        RANKING_METHOD_PROFILE_MAPPING = {
            RankingMethod.RRF: common.RANK_PROFILE_HYBRID_RRF,
            RankingMethod.NormalizeLinear: common.RANK_PROFILE_HYBRID_NORMALIZE_LINEAR,
            RankingMethod.Tensor: common.RANK_PROFILE_EMBEDDING_SIMILARITY,
            RankingMethod.Lexical: common.RANK_PROFILE_BM25
        }

        for index in [self.structured_text_index_single_field]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "doc1", "text_field_1": "dogs"},
                            {"_id": "doc2", "text_field_1": "puppies"},     # similar semantics to dogs
                            {"_id": "doc3", "text_field_1": "canines"},     # similar semantics to dogs
                            {"_id": "doc4", "text_field_1": "hot dogs"},    # shares lexical token with dogs
                            {"_id": "doc5", "text_field_1": "dogs is a word"},
                        ],
                    )
                )

                for retrieval_method in RetrievalMethod:
                    for ranking_method in RankingMethod:
                        with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                            original_query = self.config.vespa_client.query
                            def pass_through_query(*arg, **kwargs):
                                return original_query(*arg, **kwargs)

                            mock_vespa_client_query = unittest.mock.MagicMock()
                            mock_vespa_client_query.side_effect = pass_through_query

                            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                            def run():
                                res = tensor_search.search(
                                    config=self.config,
                                    index_name=index.name,
                                    text="dogs",
                                    search_method="HYBRID",
                                    hybrid_parameters=HybridParameters(
                                        retrieval_method=retrieval_method,
                                        ranking_method=ranking_method,
                                        alpha=0.6,
                                    )
                                )
                                return res

                            res = run()

                            call_args = mock_vespa_client_query.call_args_list
                            self.assertEqual(len(call_args), 1)

                            # Make sure _hybrid_search is called
                            # Check the query (yql) and ranking profile selected are as intended
                            vespa_query_kwargs = call_args[0].kwargs
                            self.assertIn(RETRIEVAL_METHOD_YQL_MAPPING[retrieval_method], vespa_query_kwargs["yql"])
                            self.assertEqual(RANKING_METHOD_PROFILE_MAPPING[ranking_method], vespa_query_kwargs["ranking"])
                            if ranking_method in {RankingMethod.RRF, RankingMethod.NormalizeLinear}:
                                self.assertEqual(0.6, vespa_query_kwargs["query_features"]["alpha"])

                            # Make sure results are retrieved
                            self.assertIn("hits", res)
                            self.assertGreater(len(res["hits"]), 0)


    def test_hybrid_search_custom_searcher(self):
        """
        Test all hybrid search methods calls the custom searcher
        """

        for index in [self.structured_text_index_single_field]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "doc1", "text_field_1": "dogs"},
                            {"_id": "doc2", "text_field_1": "puppies"},     # similar semantics to dogs
                            {"_id": "doc3", "text_field_1": "canines"},     # similar semantics to dogs
                            {"_id": "doc4", "text_field_1": "hot dogs"},    # shares lexical token with dogs
                            {"_id": "doc5", "text_field_1": "dogs is a word"},
                        ],
                    )
                )

                for retrieval_method in {RetrievalMethod.Disjunction}:
                    for ranking_method in {RankingMethod.RRF}:
                        with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                            original_query = self.config.vespa_client.query
                            def pass_through_query(*arg, **kwargs):
                                return original_query(*arg, **kwargs)

                            mock_vespa_client_query = unittest.mock.MagicMock()
                            mock_vespa_client_query.side_effect = pass_through_query

                            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                            def run():
                                res = tensor_search.search(
                                    config=self.config,
                                    index_name=index.name,
                                    text="dogs",
                                    search_method="HYBRID",
                                    hybrid_parameters=HybridParameters(
                                        retrieval_method=retrieval_method,
                                        ranking_method=ranking_method,
                                        alpha=0.6,
                                    )
                                )
                                return res

                            res = run()

                            call_args = mock_vespa_client_query.call_args_list
                            self.assertEqual(len(call_args), 1)

                            # Make sure results are retrieved
                            self.assertIn("hits", res)
                            self.assertGreater(len(res["hits"]), 0)
