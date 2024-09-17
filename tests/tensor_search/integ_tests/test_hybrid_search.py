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
from tests.marqo_test import MarqoTestCase, TestImageUrls
from marqo import exceptions as base_exceptions
import unittest
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search import api
import numpy as np
from marqo.tensor_search.models.api_models import CustomVectorQuery


class TestHybridSearch(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='sentence-transformers/all-MiniLM-L6-v2')
        )

        unstructured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True
        )

        unstructured_index_with_no_model = cls.unstructured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
            normalize_embeddings=False
        )

        unstructured_index_2_10 = cls.unstructured_marqo_index_request(
            marqo_version="2.10.0"
        )

        # STRUCTURED indexes
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

        structured_text_index_score_modifiers = cls.structured_marqo_index_request(
            model=Model(name="sentence-transformers/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
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
            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
        )

        structured_index_with_no_model = cls.structured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
                FieldRequest(name="custom_field_1", type=FieldType.CustomVector)
            ],
            tensor_fields=["text_field_1", "image_field_1", "custom_field_1"],
            normalize_embeddings=False
        )

        structured_index_empty = cls.structured_marqo_index_request(
            model=Model(name="sentence-transformers/all-MiniLM-L6-v2"),
            fields=[],
            tensor_fields=[]
        )

        structured_index_2_9 = cls.structured_marqo_index_request(
            marqo_version="2.9.0",
            fields=[],
            tensor_fields=[]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            unstructured_default_image_index,
            unstructured_index_with_no_model,
            unstructured_index_2_10,
            structured_default_image_index,
            structured_text_index_score_modifiers,
            structured_index_with_no_model,
            structured_index_empty,
            structured_index_2_9
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.unstructured_default_image_index = cls.indexes[1]
        cls.unstructured_index_with_no_model = cls.indexes[2]
        cls.unstructured_index_2_10 = cls.indexes[3]

        cls.structured_default_image_index = cls.indexes[4]
        cls.structured_text_index_score_modifiers = cls.indexes[5]
        cls.structured_index_with_no_model = cls.indexes[6]
        cls.structured_index_empty = cls.indexes[7]
        cls.structured_index_2_9 = cls.indexes[8]

    def setUp(self) -> None:
        super().setUp()

        self.docs_list = [
            # similar semantics to dogs
            {"_id": "doc1", "text_field_1": "dogs"},
            {"_id": "doc2", "text_field_1": "puppies"},
            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
            {"_id": "doc4", "text_field_1": "huskies"},
            {"_id": "doc5", "text_field_1": "four-legged animals"},

            # shares lexical token with dogs
            {"_id": "doc6", "text_field_1": "hot dogs"},
            {"_id": "doc7", "text_field_1": "dogs is a word"},
            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
            {"_id": "doc9", "text_field_1": "dogs random words"},
            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
            {"_id": "doc13", "text_field_2": "canines canines"},
        ]

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

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
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
                            retrievalMethod="disjunction",
                            rankingMethod="rrf",
                            alpha=0.6,
                            rrfK=61,
                            searchableAttributesLexical=["text_field_1", "text_field_2"] \
                                if isinstance(index, StructuredMarqoIndex) else None,
                            searchableAttributesTensor=["text_field_2", "text_field_3"] \
                                if isinstance(index, StructuredMarqoIndex) else None,
                            scoreModifiersLexical={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 1.0},
                                    {"field_name": "mult_field_2", "weight": 1.0}
                                ]
                            },
                            scoreModifiersTensor={
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
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.retrievalMethod"], RetrievalMethod.Disjunction)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rankingMethod"], RankingMethod.RRF)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.alpha"], 0.6)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rrf_k"], 61)

                self.assertEqual(vespa_query_kwargs["ranking"], "hybrid_custom_searcher")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.lexical.lexical"], "bm25")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.tensor.tensor"], "embedding_similarity")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.lexical.tensor"], "hybrid_bm25_then_embedding_similarity")
                self.assertEqual(vespa_query_kwargs["marqo__ranking.tensor.lexical"], "hybrid_embedding_similarity_then_bm25")

                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__mult_weights_lexical"],
                                 {'mult_field_1': 1.0, 'mult_field_2': 1.0})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__add_weights_lexical"], {})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__mult_weights_tensor"],
                                 {'mult_field_1': 1.0})
                self.assertEqual(vespa_query_kwargs["query_features"]["marqo__add_weights_tensor"],
                                 {'add_field_1': 1.0})

                if isinstance(index, StructuredMarqoIndex):
                    self.assertIn("(({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                                  "nearestNeighbor(marqo__embeddings_text_field_2, marqo__query_embedding)) OR "
                                  "({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                                  "nearestNeighbor(marqo__embeddings_text_field_3, marqo__query_embedding)))",
                                  vespa_query_kwargs["marqo__yql.tensor"])
                    self.assertIn(
                        "marqo__lexical_text_field_1 contains \"dogs\" OR marqo__lexical_text_field_2 contains \"dogs\"",
                        vespa_query_kwargs["marqo__yql.lexical"])
                    self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_rank_lexical"],
                                     {'marqo__lexical_text_field_1': 1, 'marqo__lexical_text_field_2': 1})
                    self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_rank_tensor"],
                                     {'marqo__embeddings_text_field_2': 1, 'marqo__embeddings_text_field_3': 1})

                elif isinstance(index, UnstructuredMarqoIndex):
                    self.assertIn("({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                                  "nearestNeighbor(marqo__embeddings, marqo__query_embedding))",
                                  vespa_query_kwargs["marqo__yql.tensor"])
                    self.assertIn(
                        "default contains \"dogs\"",
                        vespa_query_kwargs["marqo__yql.lexical"])
                    self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_rank_lexical"], {})
                    self.assertEqual(vespa_query_kwargs["query_features"]["marqo__fields_to_rank_tensor"], {})

                # TODO: For lexical/tensor and tensor/lexical. Check fields to rank specifically.
                # TODO: with and without score modifiers
                # Make sure results are retrieved
                self.assertIn("hits", res)

    def test_hybrid_search_structured_with_custom_vector_query(self):
        """
        Tests that using a custom vector query sends the correct arguments to vespa
        Structured. Uses searchable attributes.
        """
        sample_vector = [0.5 for _ in range(16)]

        original_query = self.config.vespa_client.query
        def pass_through_query(*arg, **kwargs):
            return original_query(*arg, **kwargs)

        mock_vespa_client_query = unittest.mock.MagicMock()
        mock_vespa_client_query.side_effect = pass_through_query

        with self.subTest("Custom vector query, with content, no context"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content= "sample",
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        searchableAttributesLexical=["text_field_1"],
                        searchableAttributesTensor=["text_field_1"]
                    ),
                    result_count=3
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector
            self.assertIn("marqo__lexical_text_field_1 contains \"sample\"", vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

        with self.subTest("Custom vector query, with context, with content"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content= "sample",
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        searchableAttributesLexical=["text_field_1"],
                        searchableAttributesTensor=["text_field_1"]
                    ),
                    context=SearchContext(**{"tensor": [{"vector": [i*2 for i in sample_vector],     # Double the sample vector
                                                                      "weight": 1}], })
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector with context
            self.assertIn("marqo__lexical_text_field_1 contains \"sample\"",
                          vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             [i*1.5 for i in sample_vector])    # Should average the query & context vectors
            self.assertIn("hits", res)

        with self.subTest("Custom vector query, no content, no context, tensor/tensor"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content=None,
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod="tensor",
                        rankingMethod="tensor",
                        searchableAttributesTensor=["text_field_1"]
                    ),
                    result_count=3
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings_text_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

    def test_hybrid_search_unstructured_with_custom_vector_query(self):
        """
        Tests that using a custom vector query sends the correct arguments to vespa
        Unstructured. Does not use searchable attributes.
        """
        sample_vector = [0.5 for _ in range(16)]

        original_query = self.config.vespa_client.query
        def pass_through_query(*arg, **kwargs):
            return original_query(*arg, **kwargs)

        mock_vespa_client_query = unittest.mock.MagicMock()
        mock_vespa_client_query.side_effect = pass_through_query

        with self.subTest("Custom vector query, with content, no context"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content= "sample",
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod="disjunction",
                        rankingMethod="rrf",
                    ),
                    result_count=3
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector
            self.assertIn("default contains \"sample\"", vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

        with self.subTest("Custom vector query, with context, with content"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content= "sample",
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod="disjunction",
                        rankingMethod="rrf",
                    ),
                    context=SearchContext(**{"tensor": [{"vector": [i*2 for i in sample_vector],     # Double the sample vector
                                                                      "weight": 1}], })
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector with context
            self.assertIn("default contains \"sample\"",
                          vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             [i*1.5 for i in sample_vector])    # Should average the query & context vectors
            self.assertIn("hits", res)

        with self.subTest("Custom vector query, no content, no context, tensor/tensor"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.unstructured_index_with_no_model.name,
                    text=CustomVectorQuery(
                        customVector=CustomVectorQuery.CustomVector(
                            content=None,
                            vector=sample_vector
                        )
                    ),
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod="tensor",
                        rankingMethod="tensor"
                    ),
                    result_count=3
                )
                return res

            res = run()

            call_args = mock_vespa_client_query.call_args_list

            vespa_query_kwargs = call_args[-1][1]
            self.assertIn("{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}"
                          "nearestNeighbor(marqo__embeddings, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

    def test_hybrid_search_disjunction_rrf_zero_alpha_same_as_lexical(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        alpha = 0.0

        is the same as a lexical search (in terms of result order).
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                add_docs_res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    ),
                )

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
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
                    self.assertEqual(hybrid_res["hits"][i]["_lexical_score"], lexical_res["hits"][i]["_score"])


    def test_hybrid_search_disjunction_rrf_one_alpha_same_as_tensor(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        alpha = 1.0

        is the same as a tensor search (in terms of result order).
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
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
                    self.assertEqual(hybrid_res["hits"][i]["_tensor_score"], tensor_res["hits"][i]["_score"])

    def test_hybrid_search_searchable_attributes(self):
        """
        Tests that searchable attributes work as expected for all methods
        TODO: Add unstructured index once searchable attributes are supported
        """

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list
                    )
                )

                with self.subTest("retrieval: disjunction, ranking: rrf"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="puppies",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            alpha=0.5,
                            verbose=True,
                            searchableAttributesLexical=["text_field_2"],
                            searchableAttributesTensor=["text_field_2"],
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(len(hybrid_res["hits"]), 3)            # Only 3 documents have text_field_2 at all
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")   # puppies puppies in text field 2
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc13")
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc11")

                with self.subTest("retrieval: lexical, ranking: tensor"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="puppies",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Tensor,
                            searchableAttributesLexical=["text_field_2"]
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(len(hybrid_res["hits"]), 1)        # Only 1 document has puppies in text_field_2. Lexical retrieval will only get this one.
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")

                with self.subTest("retrieval: tensor, ranking: lexical"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="puppies",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Lexical,
                            searchableAttributesTensor=["text_field_2"]
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(len(hybrid_res["hits"]), 3)    # Only 3 documents have text field 2. Tensor retrieval will get them all.
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc12")
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc11")
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc13")

    def test_hybrid_search_score_modifiers(self):
        """
        Tests that score modifiers work as expected for all methods
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "doc6", "text_field_1": "HELLO WORLD"},
                            {"_id": "doc7", "text_field_1": "HELLO WORLD", "add_field_1": 1.0},  # third
                            {"_id": "doc8", "text_field_1": "HELLO WORLD", "mult_field_1": 2.0},   # second highest score
                            {"_id": "doc9", "text_field_1": "HELLO WORLD", "mult_field_1": 3.0},  # highest score
                            {"_id": "doc10", "text_field_1": "HELLO WORLD", "mult_field_2": 3.0},    # lowest score
                        ],
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.subTest("retrieval: lexical, ranking: tensor"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Tensor,
                            scoreModifiersTensor={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                            verbose=True
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")      # highest score (score*10*3)
                    self.assertEqual(hybrid_res["hits"][0]["_score"], 30.0)
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")      # (score*10*2)
                    self.assertEqual(hybrid_res["hits"][1]["_score"], 20.0)
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")      # (score + 5*1)
                    self.assertEqual(hybrid_res["hits"][2]["_score"], 6.0)
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")      # (score)
                    self.assertEqual(hybrid_res["hits"][3]["_score"], 1.0)
                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")    # lowest score (score*-10*3)
                    self.assertEqual(hybrid_res["hits"][-1]["_score"], -30.0)

                with self.subTest("retrieval: tensor, ranking: lexical"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Lexical,
                            scoreModifiersLexical={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 2}
                                ]
                            },
                            verbose=True
                        ),
                        result_count=10,
                    )
                    self.assertIn("hits", hybrid_res)

                    base_lexical_score = hybrid_res["hits"][3]["_score"]
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")  # highest score (score*10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][0]["_score"], base_lexical_score * 10 * 3)
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # second highest score (score*10*2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_score"], base_lexical_score * 10 * 2)
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # third highest score (score + 2*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_score"], base_lexical_score + 2*1)
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")  # ORIGINAL SCORE
                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_score"], base_lexical_score * -10 * 3)

                with self.subTest("retrieval: disjunction, ranking: rrf"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="HELLO WORLD",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            scoreModifiersLexical={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                            scoreModifiersTensor={
                                "multiply_score_by": [
                                    {"field_name": "mult_field_1", "weight": 10},
                                    {"field_name": "mult_field_2", "weight": -10}
                                ],
                                "add_to_score": [
                                    {"field_name": "add_field_1", "weight": 5}
                                ]
                            },
                            verbose=True
                        ),
                        result_count=10
                    )
                    self.assertIn("hits", hybrid_res)

                    # Score without score modifiers
                    self.assertEqual(hybrid_res["hits"][3]["_id"], "doc6")  # (score)
                    base_lexical_score = hybrid_res["hits"][3]["_lexical_score"]
                    base_tensor_score = hybrid_res["hits"][3]["_tensor_score"]

                    self.assertEqual(hybrid_res["hits"][0]["_id"], "doc9")  # highest score (score*10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][0]["_lexical_score"], base_lexical_score * 10 * 3)
                    self.assertEqual(hybrid_res["hits"][0]["_tensor_score"], base_tensor_score * 10 * 3)

                    self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # (score*10*2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_lexical_score"], base_lexical_score * 10 * 2)
                    self.assertAlmostEqual(hybrid_res["hits"][1]["_tensor_score"], base_tensor_score * 10 * 2)

                    self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # (score + 5*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_lexical_score"], base_lexical_score + 5*1)
                    self.assertAlmostEqual(hybrid_res["hits"][2]["_tensor_score"], base_tensor_score + 5*1)

                    self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_lexical_score"], base_lexical_score * -10 * 3)
                    self.assertAlmostEqual(hybrid_res["hits"][-1]["_tensor_score"], base_tensor_score * -10 * 3)


    def test_hybrid_search_lexical_tensor_with_lexical_score_modifiers_succeeds(self):
        """
        Tests that if we do hybrid search with lexical retrieval and tensor ranking, we can use both lexical and tensor
        score modifiers.

        The lexical score modifiers should affect the actual result set, while the tensor score modifiers should
        affect the order and score.
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "doc4", "text_field_1": "HELLO WORLD",
                             "mult_field_1": 0.5, "add_field_1": 20},                               # OUT (negative)
                            {"_id": "doc5", "text_field_1": "HELLO WORLD", "mult_field_1": 1.0},    # OUT (negative)
                            {"_id": "doc6", "text_field_1": "HELLO WORLD"},                         # Top result
                            {"_id": "doc7", "text_field_1": "HELLO WORLD", "add_field_1": 1.0},     # Top result
                            {"_id": "doc8", "text_field_1": "HELLO WORLD", "mult_field_1": 2.0},    # OUT (negative)
                            {"_id": "doc9", "text_field_1": "HELLO WORLD", "mult_field_1": 3.0},    # OUT (negative)
                            {"_id": "doc10", "text_field_1": "HELLO WORLD", "mult_field_2": 3.0},   # Top result
                        ],
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) \
                        else None
                    )
                )

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="HELLO WORLD",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Tensor,
                        scoreModifiersLexical={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": -10},  # Will bring down doc8 and doc9. Keep doc6, doc7, doc10
                            ]
                        },
                        scoreModifiersTensor={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 10},
                                {"field_name": "mult_field_2", "weight": -10}
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 5}
                            ]
                        },
                        verbose=True
                    ),
                    result_count=3
                )
                self.assertIn("hits", hybrid_res)
                self.assertEqual(hybrid_res["hits"][0]["_id"], "doc7")      # (score + 5*1)
                self.assertEqual(hybrid_res["hits"][0]["_score"], 6.0)
                self.assertEqual(hybrid_res["hits"][1]["_id"], "doc6")      # (score)
                self.assertEqual(hybrid_res["hits"][1]["_score"], 1.0)
                self.assertEqual(hybrid_res["hits"][2]["_id"], "doc10")     # (score*-10*3)
                self.assertEqual(hybrid_res["hits"][2]["_score"], -30.0)


    def test_hybrid_search_same_retrieval_and_ranking_matches_original_method(self):
        """
        Tests that hybrid search with:
        retrieval_method = "lexical", ranking_method = "lexical" and
        retrieval_method = "tensor", ranking_method = "tensor"

        Results must be the same as lexical search and tensor search respectively.
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
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
                                retrievalMethod=retrieval_method,
                                rankingMethod=ranking_method,
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
                            self.assertEqual(hybrid_res["hits"][i]["_score"], base_res["hits"][i]["_score"])

    def test_hybrid_search_with_filter(self):
        """
        Tests that filter is applied correctly in hybrid search.
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                test_cases = [
                    (RetrievalMethod.Disjunction, RankingMethod.RRF),
                    (RetrievalMethod.Lexical, RankingMethod.Lexical),
                    (RetrievalMethod.Tensor, RankingMethod.Tensor)
                ]

                for retrieval_method, ranking_method in test_cases:
                    with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                        hybrid_res = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs something",
                            search_method="HYBRID",
                            filter="text_field_1:(something something dogs)",
                            hybrid_parameters=HybridParameters(
                                retrievalMethod=retrieval_method,
                                rankingMethod=ranking_method,
                                verbose=True
                            ),
                            result_count=10
                        )

                        self.assertIn("hits", hybrid_res)
                        self.assertEqual(len(hybrid_res["hits"]), 1)
                        self.assertEqual(hybrid_res["hits"][0]["_id"], "doc8")

    def test_hybrid_search_with_images(self):
        """
        Tests that hybrid search is accurate with images, both in query and in documents.
        Note: For unstructured, both image and link are indexed, thus the doc will have a lexical and tensor score.
        For structured, only the image is indexed, thus the doc will ONLY have a tensor score.
        """

        for index in [self.structured_default_image_index, self.unstructured_default_image_index]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "hippo image", "image_field_1": TestImageUrls.HIPPO_REALISTIC.value},
                            {"_id": "random image", "image_field_1": TestImageUrls.IMAGE2.value},
                            {"_id": "hippo text", "text_field_1": "hippo"},
                            {"_id": "hippo text low relevance", "text_field_1": "hippo text text random"},
                            {"_id": "random text", "text_field_1": "random text"}
                        ],
                        tensor_fields=["text_field_1", "image_field_1"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.subTest("disjunction text search"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="hippo",
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod="disjunction",
                            rankingMethod="rrf",
                            verbose=True
                        ),
                        result_count=4
                    )

                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "hippo text")
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "hippo text low relevance")

                with self.subTest("disjunction image search"):
                    hybrid_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text=TestImageUrls.HIPPO_REALISTIC.value,
                        search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            retrievalMethod="disjunction",
                            rankingMethod="rrf",
                            verbose=True
                        ),
                        result_count=4
                    )

                    self.assertIn("hits", hybrid_res)
                    self.assertEqual(hybrid_res["hits"][0]["_id"], "hippo image")
                    self.assertEqual(hybrid_res["hits"][1]["_id"], "random image")
                    self.assertEqual(hybrid_res["hits"][2]["_id"], "hippo text")

    def test_hybrid_search_structured_opposite_retrieval_and_ranking(self):
        """
        Tests that hybrid search with:
        retrievalMethod = "lexical", rankingMethod = "tensor" and
        retrievalMethod = "tensor", rankingMethod = "lexical"

        have expected results. The documents themselves should exactly match retrieval method, but the scores
        should match the ranking method. This is only consistent for single-field search, as retrieval top k will
        match the ranking top k.

        Uses structured index, so we have searchable attributes.
        """

        # Add documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.structured_text_index_score_modifiers.name,
                docs=self.docs_list
            )
        )

        # Reference results
        tensor_res_all_docs = tensor_search.search(  # To get tensor scores of every doc, for reference
            config=self.config,
            index_name=self.structured_text_index_score_modifiers.name,
            text="dogs",
            search_method="TENSOR",
            searchable_attributes=["text_field_1"],
            result_count=20,
        )
        lexical_res = tensor_search.search(
            config=self.config,
            index_name=self.structured_text_index_score_modifiers.name,
            text="dogs",
            search_method="LEXICAL",
            searchable_attributes=["text_field_1"],
            result_count=10,
        )
        tensor_res = tensor_search.search(
            config=self.config,
            index_name=self.structured_text_index_score_modifiers.name,
            text="dogs",
            search_method="TENSOR",
            result_count=10,
            searchable_attributes=["text_field_1"]
        )

        # Lexical retrieval with Tensor ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Lexical, rankingMethod=RankingMethod.Tensor):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.structured_text_index_score_modifiers.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Tensor,
                    searchableAttributesLexical=["text_field_1"],
                    searchableAttributesTensor=["text_field_1"],
                    verbose=True
                ),
                result_count=10
            )
            self.assertIn("hits", hybrid_res)

            # RETRIEVAL: 10 documents must match the 10 from lexical search (order may differ)
            self.assertEqual(len(hybrid_res["hits"]), len(lexical_res["hits"]))
            lexical_res_ids = [doc["_id"] for doc in lexical_res["hits"]]
            for hybrid_hit in hybrid_res["hits"]:
                self.assertIn(hybrid_hit["_id"], lexical_res_ids)

            # RANKING: scores must match the tensor search scores
            for hybrid_hit in hybrid_res["hits"]:
                tensor_hit = next(doc for doc in tensor_res_all_docs["hits"] if doc["_id"] == hybrid_hit["_id"])
                self.assertEqual(hybrid_hit["_score"], tensor_hit["_score"])

        # Tensor retrieval with Lexical ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Tensor, rankingMethod=RankingMethod.Lexical):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.structured_text_index_score_modifiers.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Lexical,
                    searchableAttributesLexical=["text_field_1"],
                    searchableAttributesTensor=["text_field_1"],
                    verbose=True
                ),
                result_count=10
            )

            self.assertIn("hits", hybrid_res)

            # RETRIEVAL: 10 documents must match the 10 from tensor search (order may differ)
            self.assertEqual(len(hybrid_res["hits"]), len(tensor_res["hits"]))
            tensor_res_ids = [doc["_id"] for doc in tensor_res["hits"]]
            for hybrid_hit in hybrid_res["hits"]:
                self.assertIn(hybrid_hit["_id"], tensor_res_ids)

            # RANKING: scores must match the lexical search scores
            for hybrid_hit in hybrid_res["hits"]:
                if hybrid_hit["_score"] > 0:
                    # Score should match its counterpart in lexical search
                    lexical_hit = next(doc for doc in lexical_res["hits"] if doc["_id"] == hybrid_hit["_id"])
                    self.assertEqual(hybrid_hit["_score"], lexical_hit["_score"])
                else:
                    # If score is 0, it should not be in lexical search results
                    self.assertNotIn(hybrid_hit["_id"], [doc["_id"] for doc in lexical_res["hits"]])

    def test_hybrid_search_unstructured_opposite_retrieval_and_ranking(self):
        """
        Tests that hybrid search with:
        retrievalMethod = "lexical", rankingMethod = "tensor" and
        retrievalMethod = "tensor", rankingMethod = "lexical"

        have expected results. The documents themselves should exactly match retrieval method, but the scores
        should match the ranking method. This is only consistent for single-field search, as retrieval top k will
        match the ranking top k.

        Uses unstructured index, so we do not use searchable attributes
        """

        # Add documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_default_image_index.name,
                docs=self.docs_list,
                tensor_fields=["text_field_1", "text_field_2"]
            )
        )

        # Reference results
        tensor_res_all_docs = tensor_search.search(  # To get tensor scores of every doc, for reference
            config=self.config,
            index_name=self.unstructured_default_image_index.name,
            text="dogs",
            search_method="TENSOR",
            result_count=20,
        )
        lexical_res = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_default_image_index.name,
            text="dogs",
            search_method="LEXICAL",
            result_count=10,
        )
        tensor_res = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_default_image_index.name,
            text="dogs",
            search_method="TENSOR",
            result_count=10,
        )

        # Lexical retrieval with Tensor ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Lexical, rankingMethod=RankingMethod.Tensor):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_image_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Tensor,
                    verbose=True
                ),
                result_count=10
            )
            self.assertIn("hits", hybrid_res)

            # RETRIEVAL: 10 documents must match the 10 from lexical search (order may differ)
            self.assertEqual(len(hybrid_res["hits"]), len(lexical_res["hits"]))
            lexical_res_ids = [doc["_id"] for doc in lexical_res["hits"]]
            for hybrid_hit in hybrid_res["hits"]:
                self.assertIn(hybrid_hit["_id"], lexical_res_ids)

            # RANKING: scores must match the tensor search scores
            for hybrid_hit in hybrid_res["hits"]:
                tensor_hit = next(doc for doc in tensor_res_all_docs["hits"] if doc["_id"] == hybrid_hit["_id"])
                self.assertEqual(hybrid_hit["_score"], tensor_hit["_score"])

        # Tensor retrieval with Lexical ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Tensor, rankingMethod=RankingMethod.Lexical):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_image_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Lexical,
                    verbose=True
                ),
                result_count=10
            )

            self.assertIn("hits", hybrid_res)

            # RETRIEVAL: 10 documents must match the 10 from tensor search (order may differ)
            self.assertEqual(len(hybrid_res["hits"]), len(tensor_res["hits"]))
            tensor_res_ids = [doc["_id"] for doc in tensor_res["hits"]]
            for hybrid_hit in hybrid_res["hits"]:
                self.assertIn(hybrid_hit["_id"], tensor_res_ids)

            # RANKING: scores must match the lexical search scores
            for hybrid_hit in hybrid_res["hits"]:
                if hybrid_hit["_score"] > 0:
                    # Score should match its counterpart in lexical search
                    lexical_hit = next(doc for doc in lexical_res["hits"] if doc["_id"] == hybrid_hit["_id"])
                    self.assertEqual(hybrid_hit["_score"], lexical_hit["_score"])
                else:
                    # If score is 0, it should not be in lexical search results
                    self.assertNotIn(hybrid_hit["_id"], [doc["_id"] for doc in lexical_res["hits"]])

    def test_hybrid_search_unstructured_highlights_for_lexical_tensor(self):
        """
        Tests that hybrid search with highlights:
        retrievalMethod = "lexical", rankingMethod = "tensor"

        has expected results and highlights even on results that have a non-tensor field.
        No highlights on the results retrieved from the non-tensor field (text_field_2 in this case),
        so list should be empty.
        """

        # Add documents
        tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.unstructured_default_image_index.name,
                docs=self.docs_list,
                tensor_fields=["text_field_1"]
            )
        )

        # Reference results
        tensor_res_all_docs = tensor_search.search(  # To get tensor scores of every doc, for reference
            config=self.config,
            index_name=self.unstructured_default_image_index.name,
            text="dogs",
            search_method="TENSOR",
            result_count=20,
        )
        lexical_res = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_default_image_index.name,
            text="dogs",
            search_method="LEXICAL",
            result_count=10,
        )

        # Lexical retrieval with Tensor ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Lexical, rankingMethod=RankingMethod.Tensor):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_image_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Tensor,
                    verbose=True
                ),
                result_count=10
            )
            self.assertIn("hits", hybrid_res)

            # RETRIEVAL: 10 documents must match the 10 from lexical search (order may differ)
            self.assertEqual(len(hybrid_res["hits"]), len(lexical_res["hits"]))
            lexical_res_ids = [doc["_id"] for doc in lexical_res["hits"]]
            for hybrid_hit in hybrid_res["hits"]:
                self.assertIn(hybrid_hit["_id"], lexical_res_ids)

            # RANKING: scores must match the tensor search scores
            for hybrid_hit in hybrid_res["hits"]:
                # All docs have highlights (text_field_1) except doc11 (from text_field_2)
                if hybrid_hit["_id"] == "doc11":
                    self.assertEqual(hybrid_hit["_highlights"], [])
                else:
                    self.assertIn("text_field_1", hybrid_hit["_highlights"][0])
                    tensor_hit = next(doc for doc in tensor_res_all_docs["hits"] if doc["_id"] == hybrid_hit["_id"])
                    self.assertEqual(hybrid_hit["_score"], tensor_hit["_score"])

    def test_hybrid_search_invalid_parameters_fails(self):
        test_cases = [
            ({
                 "alpha": 0.6,
                 "rankingMethod": "tensor"
             }, "can only be defined for 'rrf'"),
            ({
                 "rrfK": 61,
                 "rankingMethod": "normalize_linear"
             }, "can only be defined for 'rrf'"),
            ({
                 "rrfK": 60.1,
             }, "must be an integer"),
            ({
                "alpha": 1.1
            }, "between 0 and 1"),
            ({
                 "rrfK": -1
             }, "greater than or equal to 0"),
            ({
                "retrievalMethod": "disjunction",
                "rankingMethod": "lexical"
            }, "rankingMethod must be: rrf"),
            ({
                 "retrievalMethod": "tensor",
                 "rankingMethod": "rrf"
             }, "rankingMethod must be: tensor or lexical"),
            ({
                 "retrievalMethod": "lexical",
                 "rankingMethod": "rrf"
             }, "rankingMethod must be: tensor or lexical"),
            # Score modifiers need to match ranking method
            ({
                 "retrievalMethod": "tensor",
                 "rankingMethod": "tensor",
                 "scoreModifiersLexical": {
                     "multiply_score_by": [
                         {"field_name": "mult_field_1", "weight": 1.0}
                     ]
                 },
             }, "can only be defined for 'lexical',"),
            ({      # tensor/lexical can only have lexical score modifiers
                 "retrievalMethod": "tensor",
                 "rankingMethod": "lexical",
                 "scoreModifiersTensor": {
                     "multiply_score_by": [
                         {"field_name": "mult_field_1", "weight": 1.0}
                     ]
                 },
             }, "can only be defined for 'tensor',"),
            ({
                 "retrievalMethod": "lexical",
                 "rankingMethod": "lexical",
                 "scoreModifiersTensor": {
                    "multiply_score_by": [
                        {"field_name": "mult_field_1", "weight": 1.0}
                    ]
                 }
             }, "can only be defined for 'tensor',"),
            # Non-existent retrieval method
            ({"retrievalMethod": "something something"},
                "not a valid enumeration member"),
            # Non-existent ranking method
            ({"rankingMethod": "something something"},
                "not a valid enumeration member")
        ]

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                if isinstance(index, StructuredMarqoIndex):
                    final_test_cases = test_cases + [
                        # Searchable attributes need to match retrieval method
                        ({
                             "retrievalMethod": "tensor",
                             "rankingMethod": "tensor",
                             "searchableAttributesLexical": ["text_field_1"]
                         }, "can only be defined for 'lexical',"),
                        ({
                             "retrievalMethod": "lexical",
                             "rankingMethod": "lexical",
                             "searchableAttributesTensor": ["text_field_1"]
                         }, "can only be defined for 'tensor',")
                    ]
                else:
                    # Unstructured hybrid search cannot have searchable attributes
                    final_test_cases = test_cases

                for hybrid_parameters, error_message in final_test_cases:
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
        Ensure that searchable_attributes and score_modifiers cannot be set in hybrid search.
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                with self.subTest("searchable_attributes active"):
                    with self.assertRaises(ValueError) as e:
                        tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method="HYBRID",
                            searchable_attributes=["text_field_1"]
                        )
                    self.assertIn("'searchableAttributes' cannot be used for hybrid", str(e.exception))

                with self.subTest("score_modifiers active"):
                    with self.assertRaises(ValueError) as e:
                        tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method="HYBRID",
                            score_modifiers=ScoreModifierLists(
                                multiply_score_by=[
                                    {"field_name": "mult_field_1", "weight": 1.0}
                                ],
                                add_to_score=[
                                    {"field_name": "add_field_1", "weight": 1.0}
                                ]
                            ),
                        )
                    self.assertIn("'scoreModifiers' cannot be used for hybrid", str(e.exception))

    def test_hybrid_search_structured_invalid_fields_fails(self):
        """
        If searching with HYBRID, searchableAttributesLexical must only have lexical fields, and
        searchableAttributesTensor must only have tensor fields.
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
                            retrievalMethod=retrieval_method,
                            rankingMethod=ranking_method,
                            searchableAttributesLexical=["text_field_1", "add_field_1"]
                        )
                    )
                self.assertIn("has no lexically searchable field add_field_1", str(e.exception))

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
                            searchableAttributesTensor=["mult_field_1", "text_field_1"]
                        )
                    )
                self.assertIn("has no tensor field mult_field_1", str(e.exception))

    def test_hybrid_search_default_parameters(self):
        """
        Test hybrid search when no hybrid parameters are provided.
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_default_text_index]:
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
                    )
                    return res

                res = run()

                call_args = mock_vespa_client_query.call_args_list
                self.assertEqual(len(call_args), 1)

                vespa_query_kwargs = call_args[0][1]
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.retrievalMethod"], RetrievalMethod.Disjunction)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rankingMethod"], RankingMethod.RRF)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.alpha"], 0.5)
                self.assertEqual(vespa_query_kwargs["marqo__hybrid.rrf_k"], 60)

                # Make sure results are retrieved
                self.assertIn("hits", res)

    def test_hybrid_search_structured_index_has_no_hybrid_rank_profile_fails(self):
        """
        If an index does not have both lexical and tensor fields, it will have no hybrid rank profile.
        If hybrid search is done on such an index, it should fail with 400.

        Tests for lexical and tensor search methods as well.
        """

        # Lexical search
        with self.subTest("lexical search"):
            with self.assertRaises(core_exceptions.InvalidArgumentError) as cm:
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_empty.name,
                    text="dogs",
                    search_method="LEXICAL",
                    result_count=10
                )
            self.assertIn("no lexically searchable fields", str(cm.exception))

        # Tensor search
        with self.subTest("tensor search"):
            with self.assertRaises(core_exceptions.InvalidArgumentError) as cm:
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_empty.name,
                    text="dogs",
                    search_method="TENSOR",
                    result_count=10
                )
            self.assertIn("no tensor fields", str(cm.exception))

        # Hybrid search
        with self.subTest("hybrid search"):
            with self.assertRaises(core_exceptions.InvalidArgumentError) as cm:
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.structured_index_empty.name,
                    text="dogs",
                    search_method="HYBRID",
                    result_count=10
                )
            self.assertIn("either has no tensor fields or no lexically searchable fields", str(cm.exception))

    def test_hybrid_search_none_query_wrong_retrieval_or_ranking_fails(self):
        """
        Test that hybrid search with a None query and wrong retrieval or ranking method fails.
        """
        custom_vector = [0.655 for _ in range(16)]
        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Lexical)
        ]

        for index in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with self.subTest(index=index.name):
                for retrieval_method, ranking_method in test_cases:
                    with self.subTest(retrieval=retrieval_method, ranking=ranking_method):
                        with self.assertRaises(InvalidArgumentError) as e:
                            tensor_search.search(
                                config=self.config,
                                index_name=index.name,
                                text=None,
                                search_method="HYBRID",
                                hybrid_parameters=HybridParameters(
                                    retrievalMethod=retrieval_method,
                                    rankingMethod=ranking_method,
                                    verbose=True
                                )
                            )
                        self.assertIn("unless retrieval_method and ranking_method are both 'tensor'", str(e.exception))

    def test_hybrid_search_none_query_with_context_vectors_passes(self):
        """Test to ensure that context vectors work with no_model by setting query as None and providing context
        vectors in search method. Uses hybrid search.
        """

        custom_vector = [0.655 for _ in range(16)]

        docs = [
            {
                "_id": "1",
                "custom_field_1":
                    {
                        "content": "test custom field content_1",
                        "vector": np.random.rand(16).tolist()
                    }
            },
            {
                "_id": "2",
                "custom_field_1":
                    {
                        "content": "test custom field content_2",
                        "vector": custom_vector
                    }
            }
        ]

        for index in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index.name)):
                add_docs_params = AddDocsParams(index_name=index.name,
                                                docs=docs,
                                                tensor_fields=["custom_field_1"] \
                                                    if isinstance(index, UnstructuredMarqoIndex) else None,
                                                mappings={"custom_field_1": {"type": "custom_vector"}} \
                                                    if isinstance(index, UnstructuredMarqoIndex) else None)
                _ = tensor_search.add_documents(config=self.config,
                                                add_docs_params=add_docs_params)

                r = tensor_search.search(config=self.config, index_name=index.name, text=None,
                                         search_method="hybrid",
                                         hybrid_parameters=HybridParameters(retrievalMethod=RetrievalMethod.Tensor,
                                                                            rankingMethod=RankingMethod.Tensor,
                                                                            verbose=True),
                                         context=SearchContext(**{"tensor": [{"vector": custom_vector,
                                                                              "weight": 1}], }))
                self.assertEqual(2, len(r["hits"]))
                self.assertEqual("2", r["hits"][0]["_id"])
                self.assertAlmostEqual(1, r["hits"][0]["_score"], places=1)

                self.assertEqual("1", r["hits"][1]["_id"])
                self.assertTrue(r["hits"][1]["_score"], r["hits"][0]["_score"])

    def test_hybrid_search_unstructured_with_searchable_attributes_fails(self):
        """
        Test that hybrid search with unstructured index and searchable attributes fails.
        TODO: Remove when this is supported
        """

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_text_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    searchableAttributesLexical=["text_field_1"]
                )
            )
        self.assertIn("does not support `searchableAttributesTensor` or `searchableAttributesLexical`",
                      str(e.exception))

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_text_index.name,
                text="dogs",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Tensor,
                    searchableAttributesTensor=["text_field_1"]
                )
            )
        self.assertIn("does not support `searchableAttributesTensor` or `searchableAttributesLexical`",
                      str(e.exception))

    def test_hybrid_search_unstructured_with_2_10_fails(self):
        """
        Test that hybrid search with unstructured index with version 2.10.0 fails (version is below the minimum).
        """

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.unstructured_index_2_10.name,
                text="dogs",
                search_method="HYBRID",
            )
        self.assertIn("only supported for Marqo unstructured indexes created with Marqo 2.11.0 or later",
                      str(e.exception))

    def test_hybrid_search_structured_with_2_9_fails(self):
        """
        Test that hybrid search with structured index with version 2.9.0 fails (version is below the minimum).
        """

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.structured_index_2_9.name,
                text="dogs",
                search_method="HYBRID",
            )
        self.assertIn("only supported for Marqo structured indexes created with Marqo 2.10.0 or later",
                      str(e.exception))

    def test_hybrid_parameters_with_wrong_search_method_fails(self):
        """
        Test that hybrid parameters with wrong search method fails.
        """

        # TODO: Use api.search() instead of tensor_search.search()
        # Covered in API tests
        pass