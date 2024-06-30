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
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.tensor_search.models.api_models import ScoreModifierLists as apiScoreModifier


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

        structured_text_index_score_modifiers = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="add_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="add_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
            ],
            tensor_fields=["text_field_1", "text_field_2"]
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
            structured_text_index_single_field,
            structured_text_index_score_modifiers
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
        cls.structured_text_index_score_modifiers = cls.indexes[10]

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
        Test all hybrid search calls the correct vespa queries.
        """

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
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
                            retrieval_method="disjunction",
                            ranking_method="rrf",
                            alpha=0.6,
                            rrf_k=61,
                            searchable_attributes_lexical=["text_field_1", "text_field_2"],
                            searchable_attributes_tensor=["text_field_2"],
                            score_modifiers_lexical={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 1.0},
                                    {"field_name": "mult_field_2", "weight": 1.0}
                                ]
                            },
                            score_modifiers_tensor={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 1.0}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 1.0}
                                ]
                            }

                        )
                    )
                    return res

                res = run()

                call_args = mock_vespa_client_query.call_args_list
                self.assertEqual(len(call_args), 1)

                vespa_query_kwargs = call_args[0][1]
                self.assertIn("PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.", vespa_query_kwargs["yql"])
                self.assertIn("(({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                              "nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding)) OR "
                              "({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                              "nearestNeighbor(marqo__embeddings_text_field_2, marqo__query_embedding)))",
                              vespa_query_kwargs["marqo__yql.tensor"])
                self.assertIn("weakAnd(default contains \"dogs\")", vespa_query_kwargs["marqo__yql.lexical"])
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.retrievalMethod"], RetrievalMethod.Disjunction)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rankingMethod"], RankingMethod.RRF)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.tensorScoreModifiersPresent"], True)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.lexicalScoreModifiersPresent"], True)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.alpha"], 0.6)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rrf_k"], 61)

                self.assertEqual(vespa_query_kwargs["ranking"], "hybrid_custom_searcher")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.lexical"], "bm25")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.tensor"], "embedding_similarity")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.lexicalScoreModifiers"], "bm25_modifiers")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.tensorScoreModifiers"], "embedding_similarity_modifiers")

                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_search_lexical"], {'text_field_1': 1, 'text_field_2': 1})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_search_tensor"], {'text_field_2': 1})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__mult_weights_lexical"], {'mult_field_1': 1.0, 'mult_field_2': 1.0})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__add_weights_lexical"], {})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__mult_weights_tensor"], {'mult_field_1': 1.0})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__add_weights_tensor"], {'add_field_1': 1.0})

                # Make sure results are retrieved
                self.assertIn("hits", res)


    def test_hybrid_search_disjunction_rrf_zero_alpha_same_as_lexical(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        alpha = 0.0

        is the same as a lexical search (in terms of result order).
        """

        for index in [self.structured_text_index_single_field]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            # similar semantics to dogs
                            {"_id": "doc1", "text_field_1": "dogs"},        # URI: 83e4b178e136f600809a80e0
                            {"_id": "doc2", "text_field_1": "puppies"},     # URI: 271559ecc987c6a0cf7768c9
                            {"_id": "doc3", "text_field_1": "canines"},
                            {"_id": "doc4", "text_field_1": "huskies"},
                            {"_id": "doc5", "text_field_1": "four-legged animals"},

                            # shares lexical token with dogs
                            {"_id": "doc6", "text_field_1": "hot dogs"},
                            {"_id": "doc7", "text_field_1": "dogs is a word"},  # URI: 84299cc1111b4e299606971f
                            {"_id": "doc8", "text_field_1": "something something dogs"},
                            {"_id": "doc9", "text_field_1": "dogs random words"},
                            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},
                        ],
                    )
                )

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrieval_method=RetrievalMethod.Disjunction,
                        ranking_method=RankingMethod.RRF,
                        alpha=0,
                        verbose=True
                    ),
                    result_count=10
                )

                lexical_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="LEXICAL",
                    result_count=10
                )

                self.assertEqual(len(hybrid_res["hits"]), len(lexical_res["hits"]))
                for i in range(len(hybrid_res["hits"])):
                    self.assertEqual(hybrid_res["hits"][i]["_id"], lexical_res["hits"][i]["_id"])

    def test_hybrid_search_disjunction_rrf_one_alpha_same_as_tensor(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        alpha = 1.0

        is the same as a tensor search (in terms of result order).
        """

        for index in [self.structured_text_index_single_field]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            # similar semantics to dogs
                            {"_id": "doc1", "text_field_1": "dogs"},        # URI: 83e4b178e136f600809a80e0
                            {"_id": "doc2", "text_field_1": "puppies"},     # URI: 271559ecc987c6a0cf7768c9
                            {"_id": "doc3", "text_field_1": "canines"},
                            {"_id": "doc4", "text_field_1": "huskies"},
                            {"_id": "doc5", "text_field_1": "four-legged animals"},

                            # shares lexical token with dogs
                            {"_id": "doc6", "text_field_1": "hot dogs"},
                            {"_id": "doc7", "text_field_1": "dogs is a word"},  # URI: 84299cc1111b4e299606971f
                            {"_id": "doc8", "text_field_1": "something something dogs"},
                            {"_id": "doc9", "text_field_1": "dogs random words"},
                            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},
                        ],
                    )
                )

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrieval_method=RetrievalMethod.Disjunction,
                        ranking_method=RankingMethod.RRF,
                        alpha=1.0,
                        verbose=True
                    ),
                    result_count=10
                )

                tensor_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="TENSOR",
                    result_count=10
                )

                self.assertEqual(len(hybrid_res["hits"]), len(tensor_res["hits"]))
                for i in range(len(hybrid_res["hits"])):
                    self.assertEqual(hybrid_res["hits"][i]["_id"], tensor_res["hits"][i]["_id"])

    def test_hybrid_search_score_modifiers_searchable_attributes(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        Using score_modifiers and searchable_attributes
        """

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            # similar semantics to dogs
                            {"_id": "doc1", "text_field_1": "dogs"},        # URI: 83e4b178e136f600809a80e0
                            {"_id": "doc2", "text_field_1": "puppies"},     # URI: 271559ecc987c6a0cf7768c9
                            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
                            {"_id": "doc4", "text_field_1": "huskies"},
                            {"_id": "doc5", "text_field_1": "four-legged animals"},

                            # shares lexical token with dogs
                            {"_id": "doc6", "text_field_1": "hot dogs"},
                            {"_id": "doc7", "text_field_1": "dogs is a word"},  # URI: 84299cc1111b4e299606971f
                            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
                            {"_id": "doc9", "text_field_1": "dogs random words"},
                            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

                            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
                            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
                            {"_id": "doc13", "text_field_2": "canines canines"},

                        ],
                    )
                )

                with self.subTest("retrieval: disjunction, ranking: rrf"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrieval_method=RetrievalMethod.Disjunction,
                            ranking_method=RankingMethod.RRF,
                            alpha=0.8,
                            searchable_attributes_lexical=["text_field_1", "text_field_2"],
                            searchable_attributes_tensor=["text_field_2"],
                            score_modifiers_lexical={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 1.0},
                                    {"field_name": "mult_field_2", "weight": 1.0}
                                ]
                            },
                            score_modifiers_tensor={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 1.0}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 1.0}
                                ]
                            },
                            verbose=True
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)

                    # self.assertEqual(len(hybrid_res["hits"]), len(tensor_res["hits"]))
                    #for i in range(len(hybrid_res["hits"])):
                    #    self.assertEqual(hybrid_res["hits"][i]["_id"], tensor_res["hits"][i]["_id"])


    def test_hybrid_search_same_retrieval_and_ranking_matches_original_method(self):
        """
        Tests that hybrid search with:
        retrieval_method = "lexical", ranking_method = "lexical" and
        retrieval_method = "tensor", ranking_method = "tensor"

        Results must be the same as lexical search and tensor search respectively.
        """

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            # similar semantics to dogs
                            {"_id": "doc1", "text_field_1": "dogs"},        # URI: 83e4b178e136f600809a80e0
                            {"_id": "doc2", "text_field_1": "puppies"},     # URI: 271559ecc987c6a0cf7768c9
                            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
                            {"_id": "doc4", "text_field_1": "huskies"},
                            {"_id": "doc5", "text_field_1": "four-legged animals"},

                            # shares lexical token with dogs
                            {"_id": "doc6", "text_field_1": "hot dogs"},
                            {"_id": "doc7", "text_field_1": "dogs is a word"},  # URI: 84299cc1111b4e299606971f
                            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
                            {"_id": "doc9", "text_field_1": "dogs random words"},
                            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

                            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
                            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
                            {"_id": "doc13", "text_field_2": "canines canines"},

                        ],
                    )
                )

                test_cases = [
                    (RetrievalMethod.Lexical, RankingMethod.Lexical),
                    (RetrievalMethod.Tensor, RankingMethod.Tensor)
                ]

                for retrieval_method, ranking_method in test_cases:
                    with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                        hybrid_res = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method="HYBRID",
                            hybrid_parameters=HybridParameters(
                                retrieval_method=retrieval_method,
                                ranking_method=ranking_method,
                                verbose=True
                            ),
                            result_count=10
                        )

                        base_res = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method=retrieval_method,     # will be either lexical or tensor
                            result_count=10
                        )

                        self.assertIn("hits", hybrid_res)
                        self.assertIn("hits", base_res)
                        self.assertEqual(len(hybrid_res["hits"]), len(base_res["hits"]))
                        for i in range(len(hybrid_res["hits"])):
                            self.assertEqual(hybrid_res["hits"][i]["_id"], base_res["hits"][i]["_id"])

    def test_hybrid_search_opposite_retrieval_and_ranking(self):
        """
        Tests that hybrid search with:
        retrieval_method = "lexical", ranking_method = "tensor" and
        retrieval_method = "tensor", ranking_method = "lexical"

        TODO: Figure out testing metric.
        """

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            # similar semantics to dogs
                            {"_id": "doc1", "text_field_1": "dogs"},        # URI: 83e4b178e136f600809a80e0
                            {"_id": "doc2", "text_field_1": "puppies"},     # URI: 271559ecc987c6a0cf7768c9
                            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
                            {"_id": "doc4", "text_field_1": "huskies"},
                            {"_id": "doc5", "text_field_1": "four-legged animals"},

                            # shares lexical token with dogs
                            {"_id": "doc6", "text_field_1": "hot dogs"},
                            {"_id": "doc7", "text_field_1": "dogs is a word"},  # URI: 84299cc1111b4e299606971f
                            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
                            {"_id": "doc9", "text_field_1": "dogs random words"},
                            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

                            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
                            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
                            {"_id": "doc13", "text_field_2": "canines canines"},

                        ],
                    )
                )

                test_cases = [
                    (RetrievalMethod.Lexical, RankingMethod.Tensor),
                    (RetrievalMethod.Tensor, RankingMethod.Lexical)
                ]

                for retrieval_method, ranking_method in test_cases:
                    with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                        hybrid_res = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method="HYBRID",
                            hybrid_parameters=HybridParameters(
                                retrieval_method=retrieval_method,
                                ranking_method=ranking_method,
                                verbose=True
                            ),
                            result_count=10
                        )

                        self.assertIn("hits", hybrid_res)

    def test_hybrid_search_invalid_parameters_fails(self):
        test_cases = [
            ({
                 "alpha": 0.6,
                 "ranking_method": "tensor"
             }, "can only be defined for 'rrf' and "),
            ({
                 "rrf_k": 61,
                 "ranking_method": "normalize_linear"
             }, "can only be defined for 'rrf'"),
            ({
                 "rrf_k": 60.1,
             }, "must be an integer"),
            ({
                "alpha": 1.1
            }, "between 0 and 1"),
            ({
                 "rrf_k": -1
             }, "greater than or equal to 0"),
            ({
                "retrieval_method": "disjunction",
                "ranking_method": "lexical"
            }, "ranking_method must be: rrf"),
            ({
                 "retrieval_method": "tensor",
                 "ranking_method": "rrf"
             }, "ranking_method must be: tensor or lexical"),
            ({
                 "retrieval_method": "lexical",
                 "ranking_method": "rrf"
             }, "ranking_method must be: tensor or lexical"),
            # Searchable attributes need to match retrieval method
            ({
                "retrieval_method": "tensor",
                "ranking_method": "tensor",
                "searchable_attributes_lexical": ["text_field_1"]
             }, "can only be defined for 'lexical',"),
            ({
                "retrieval_method": "lexical",
                "ranking_method": "lexical",
                "searchable_attributes_tensor": ["text_field_1"]
             }, "can only be defined for 'tensor',"),
            # Score modifiers need to match ranking method
            ({
                 "retrieval_method": "tensor",
                 "ranking_method": "tensor",
                 "score_modifiers_lexical": {
                     "multiply_score_by": [
                         {"field_name": "mult_field_1", "weight": 1.0}
                     ]
                 },
             }, "can only be defined for 'lexical',"),
            ({
                 "retrieval_method": "lexical",
                 "ranking_method": "lexical",
                 "score_modifiers_tensor": {
                    "multiply_score_by": [
                        {"field_name": "mult_field_1", "weight": 1.0}
                    ]
                 }
             }, "can only be defined for 'tensor',"),
            # Non-existent retrieval method
            ({"retrieval_method": "something something"},
                "not a valid enumeration member"),
            # Non-existent ranking method
            ({"ranking_method": "something something"},
                "not a valid enumeration member")
        ]
        # TODO: add unstructured index
        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                for hybrid_parameters, error_message in test_cases:
                    with self.subTest(hybrid_parameters=hybrid_parameters):
                        with self.assertRaises(ValueError) as e:
                            tensor_search.search(
                                config=self.config,
                                index_name=index.name,
                                text="dogs",
                                search_method="HYBRID",
                                hybrid_parameters=HybridParameters(**hybrid_parameters)
                            )
                        self.assertIn(error_message, str(e.exception))

    def test_hybrid_search_conflicting_parameters_fails(self):
        """
        Ensure that searchable_attributes cannot be set alongside hybrid_parameters.searchable_attributes_lexical or
        hybrid_parameters.searchable_attributes_tensor.

        score_modifiers cannot be set alongside hybrid_parameters.score_modifiers_lexical or
        hybrid_parameters.score_modifiers_tensor.
        """

        test_cases = [
            ({
                 "searchable_attributes_lexical": ["text_field_1"]
            }, "or searchable_attributes, not both."),
            ({
                 "searchable_attributes_tensor": ["text_field_1"]
            }, "or searchable_attributes, not both."),
            ({
                 "score_modifiers_lexical": {
                    "multiply_score_by": [
                        {"field_name": "mult_field_1", "weight": 1.0}
                    ]
                 }
             }, "or score_modifiers, not both."),
            ({
                 "score_modifiers_tensor": {
                    "multiply_score_by": [
                        {"field_name": "mult_field_1", "weight": 1.0}
                    ]
                 }
             }, "or score_modifiers, not both."),
        ]
        # TODO: add unstructured index
        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                for hybrid_parameters, error_message in test_cases:
                    with self.subTest(hybrid_parameters=hybrid_parameters):
                        with self.assertRaises(ValueError) as e:
                            tensor_search.search(
                                config=self.config,
                                index_name=index.name,
                                text="dogs",
                                search_method="HYBRID",
                                score_modifiers=apiScoreModifier(
                                    multiply_score_by=[
                                        {"field_name": "mult_field_1", "weight": 1.0}
                                    ],
                                    add_to_score=[
                                        {"field_name": "add_field_1", "weight": 1.0}
                                    ]
                                ),
                                searchable_attributes=["text_field_1"],
                                hybrid_parameters=HybridParameters(**hybrid_parameters)
                            )
                        self.assertIn(error_message, str(e.exception))

    def test_hybrid_search_structured_invalid_fields_fails(self):
        """
        If searching with HYBRID, searchable_attributes_lexical must only have lexical fields, and
        searchable_attributes_tensor must only have tensor fields.
        """
        # Non-lexical field
        test_cases = [
            ("disjunction", "rrf"),
            ("lexical", "lexical"),
            ("lexical", "tensor")
        ]
        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                with self.assertRaises(core_exceptions.InvalidFieldNameError) as e:
                    tensor_search.search(
                        config=self.config,
                        index_name=self.structured_text_index_score_modifiers.name,
                        text="dogs",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrieval_method=retrieval_method,
                            ranking_method=ranking_method,
                            searchable_attributes_lexical=["text_field_1", "add_field_1"]
                        )
                    )
                self.assertIn("has no lexical field add_field_1", str(e.exception))

        # Non-tensor field
        test_cases = [
            ("disjunction", "rrf"),
            ("tensor", "tensor"),
            ("tensor", "lexical")
        ]
        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                with self.assertRaises(core_exceptions.InvalidFieldNameError) as e:
                    tensor_search.search(
                        config=self.config,
                        index_name=self.structured_text_index_score_modifiers.name,
                        text="dogs",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            searchable_attributes_tensor=["mult_field_1", "text_field_1"]
                        )
                    )
                self.assertIn("has no tensor field mult_field_1", str(e.exception))

    # TODO: Remove when unstructured index is supported
    @unittest.skip
    def test_hybrid_search_on_unstructured_index_fails(self):
        """
        Test that hybrid search on an unstructured index fails.
        """

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_text_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrieval_method=RetrievalMethod.Disjunction,
                    ranking_method=RankingMethod.RRF,
                    alpha=0.6
                )
            )
        self.assertIn("not yet supported for hybrid search", str(e.exception))
