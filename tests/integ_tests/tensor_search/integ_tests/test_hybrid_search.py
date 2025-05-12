import copy
import json
import os
import unittest
from unittest import mock

import httpx
import numpy as np
from fastapi.responses import ORJSONResponse
from integ_tests.marqo_test import MarqoTestCase, TestImageUrls, EXAMPLE_FASHION_DOCUMENTS

import marqo.core.exceptions as core_exceptions
import marqo.vespa.exceptions as vespa_exceptions
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import CustomVectorQuery
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration, RangeConfiguration
import pytest

import unittest


class TestHybridSearch(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        semi_structured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True
        )

        semi_structured_index_with_no_model = cls.unstructured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
            normalize_embeddings=False
        )

        semi_structured_text_index_2_14 = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            marqo_version='2.14.0'
        )

        # Legacy UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            marqo_version='2.12.0'
        )

        unstructured_default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion400m_e31'),  # Used to be ViT-B/32 in old structured tests
            treat_urls_and_pointers_as_images=True,
            marqo_version='2.12.0'
        )

        unstructured_index_with_no_model = cls.unstructured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
            normalize_embeddings=False,
            marqo_version='2.12.0'
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
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_tensor_only", type=FieldType.Text, features=[]),
                FieldRequest(name="text_lexical_only", type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="add_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="add_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "text_field_3", "text_tensor_only"]
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
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[],
            tensor_fields=[]
        )

        structured_index_2_9 = cls.structured_marqo_index_request(
            marqo_version="2.9.0",
            fields=[],
            tensor_fields=[]
        )

        structured_text_index_2_14 = cls.structured_marqo_index_request(
            marqo_version="2.14.0",
            fields=[FieldRequest(name='text_field_1', type=FieldType.Text, features=[FieldFeature.LexicalSearch])],
            tensor_fields=["text_field_1"]
        )

        structured_index_one_tensor_field = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
            ],
            tensor_fields=["text_field_1"]
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
            structured_index_2_9,
            structured_text_index_2_14,
            structured_index_one_tensor_field,
            semi_structured_default_text_index,
            semi_structured_default_image_index,
            semi_structured_index_with_no_model,
            semi_structured_text_index_2_14
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
        cls.structured_text_index_2_14 = cls.indexes[9]
        cls.structured_index_one_tensor_field = cls.indexes[10]

        cls.semi_structured_default_text_index = cls.indexes[11]
        cls.semi_structured_default_image_index = cls.indexes[12]
        cls.semi_structured_index_with_no_model = cls.indexes[13]
        cls.semi_structured_text_index_2_14 = cls.indexes[14]

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

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                original_query = self.config.vespa_client.query
                def pass_through_query(*arg, **kwargs):
                    return original_query(*arg, **kwargs)

                mock_vespa_client_query = unittest.mock.MagicMock()
                mock_vespa_client_query.side_effect = pass_through_query

                if isinstance(index, UnstructuredMarqoIndex):
                    # this is required to create the tensor fields in the semi-structured index
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=[{"_id": "1", "text_field_1": "dogs", "text_field_2": "cats", "text_field_3": "cows"},],
                            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
                        ),
                    )

                @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
                def run():
                    res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        rerank_depth=None if index == self.unstructured_default_text_index else 3,
                        result_count=3,
                        hybrid_parameters=HybridParameters(
                            retrievalMethod="disjunction",
                            rankingMethod="rrf",
                            alpha=0.6,
                            rrfK=61,
                            searchableAttributesLexical=["text_field_1", "text_field_2"] \
                                if isinstance(index, (StructuredMarqoIndex, SemiStructuredMarqoIndex)) else None,
                            searchableAttributesTensor=["text_field_2", "text_field_3"] \
                                if isinstance(index, (StructuredMarqoIndex, SemiStructuredMarqoIndex)) else None,
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
                self.assertEqual(vespa_query_kwargs["hits"], 3)

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

                if isinstance(index, (StructuredMarqoIndex, SemiStructuredMarqoIndex)):
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
                    # global rerankDepth & score modifiers are not available for legacy unstructured
                    self.assertEqual(vespa_query_kwargs["marqo__hybrid.rerankDepthGlobal"], 3)

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

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

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

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

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

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

    def test_hybrid_search_semi_structured_with_custom_vector_query(self):
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

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_index_with_no_model.name,
                docs=[{"_id": "doc1", "custom_field_1":
                    {
                        "content": "test custom field content_1",
                        "vector": np.random.rand(16).tolist()
                    }}],
                tensor_fields=["custom_field_1"],
                mappings={"custom_field_1": {"type": "custom_vector"}}
            ),
        )

        with self.subTest("Custom vector query, with content, no context"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.semi_structured_index_with_no_model.name,
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
                          "nearestNeighbor(marqo__embeddings_custom_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector
            self.assertIn("default contains \"sample\"", vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

        with self.subTest("Custom vector query, with context, with content"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.semi_structured_index_with_no_model.name,
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
                          "nearestNeighbor(marqo__embeddings_custom_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            # Lexical yql has content text, but Tensor uses the sample vector with context
            self.assertIn("default contains \"sample\"",
                          vespa_query_kwargs["marqo__yql.lexical"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             [i*1.5 for i in sample_vector])    # Should average the query & context vectors
            self.assertIn("hits", res)

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

        with self.subTest("Custom vector query, no content, no context, tensor/tensor"):
            @unittest.mock.patch("marqo.vespa.vespa_client.VespaClient.query", mock_vespa_client_query)
            def run():
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.semi_structured_index_with_no_model.name,
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
                          "nearestNeighbor(marqo__embeddings_custom_field_1, marqo__query_embedding)",
                          vespa_query_kwargs["marqo__yql.tensor"])
            self.assertEqual(vespa_query_kwargs["query_features"]["marqo__query_embedding"],
                             sample_vector)
            self.assertIn("hits", res)

            # Result should be JSON serializable
            try:
                ORJSONResponse(res)
            except TypeError as e:
                self.fail(f"Result is not JSON serializable: {e}")

    def test_hybrid_search_disjunction_rrf_zero_alpha_same_as_lexical(self):
        """
        Tests that hybrid search with:
        retrieval_method = "disjunction"
        ranking_method = "rrf"
        alpha = 0.0

        is the same as a lexical search (in terms of result order).
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                add_docs_res = self.add_documents(
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

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                self.add_documents(
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
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
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
                    # doc11 and doc13 has score 0, so their order is non-deterministic
                    self.assertSetEqual({'doc11', 'doc13'}, {hit["_id"] for hit in hybrid_res["hits"][1:]})

    def test_hybrid_search_score_modifiers_different_retrieval_and_ranking(self):
        """
        Tests that score modifiers work as expected for tensor/lexical and lexical/tensor hybrid
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

    def test_hybrid_search_all_score_modifiers_fusion(self):
        """
        Tests that all score modifiers tensor, lexical, and global, work as expected together.
        They should all work independently of each other. tensor, lexical, and _score should all be modified
        Tests all fusion methods (currently only RRF / Disjunction)
        Does not test legacy unstructured index, as it does not support global score modifiers (2.15 onwards)
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

                # Calculate unmodified RRF scores
                hybrid_res_no_global_score_modifiers = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="HELLO WORLD",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        # Mult weight of 1000 used so it outweighs the add weight of 5
                        scoreModifiersLexical={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1000},
                                {"field_name": "mult_field_2", "weight": -1000}
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 5}
                            ]
                        },
                        scoreModifiersTensor={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1000},
                                {"field_name": "mult_field_2", "weight": -1000}
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 5}
                            ]
                        },
                        verbose=True
                    ),
                    result_count=10
                )

                # RRF scores without global score modifiers
                unmodified_rrf_scores = {}
                for hit in hybrid_res_no_global_score_modifiers["hits"]:
                    unmodified_rrf_scores[hit["_id"]] = hit["_score"]

                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="HELLO WORLD",
                    search_method="HYBRID",
                    score_modifiers=ScoreModifierLists(
                        multiply_score_by=[
                            {"field_name": "mult_field_1", "weight": 1000},
                            {"field_name": "mult_field_2", "weight": -1000}
                        ],
                        add_to_score=[
                            {"field_name": "add_field_1", "weight": 5}
                        ]
                    ),
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        # Mult weight of 1000 used so it outweighs the add weight of 5
                        scoreModifiersLexical={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1000},
                                {"field_name": "mult_field_2", "weight": -1000}
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 5}
                            ]
                        },
                        scoreModifiersTensor={
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1000},
                                {"field_name": "mult_field_2", "weight": -1000}
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
                self.assertAlmostEqual(hybrid_res["hits"][0]["_lexical_score"], base_lexical_score * 1000 * 3)
                self.assertAlmostEqual(hybrid_res["hits"][0]["_tensor_score"], base_tensor_score * 1000 * 3)
                self.assertAlmostEqual(hybrid_res["hits"][0]["_score"], unmodified_rrf_scores["doc9"] * 1000 * 3)

                self.assertEqual(hybrid_res["hits"][1]["_id"], "doc8")  # (score*10*2)
                self.assertAlmostEqual(hybrid_res["hits"][1]["_lexical_score"], base_lexical_score * 1000 * 2)
                self.assertAlmostEqual(hybrid_res["hits"][1]["_tensor_score"], base_tensor_score * 1000 * 2)
                self.assertAlmostEqual(hybrid_res["hits"][1]["_score"], unmodified_rrf_scores["doc8"] * 1000 * 2)

                self.assertEqual(hybrid_res["hits"][2]["_id"], "doc7")  # (score + 5*1)
                self.assertAlmostEqual(hybrid_res["hits"][2]["_lexical_score"], base_lexical_score + 5*1)
                self.assertAlmostEqual(hybrid_res["hits"][2]["_tensor_score"], base_tensor_score + 5*1)
                self.assertAlmostEqual(hybrid_res["hits"][2]["_score"], unmodified_rrf_scores["doc7"] + 5*1)

                self.assertEqual(hybrid_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-10*3)
                self.assertAlmostEqual(hybrid_res["hits"][-1]["_lexical_score"], base_lexical_score * -1000 * 3)
                self.assertAlmostEqual(hybrid_res["hits"][-1]["_tensor_score"], base_tensor_score * -1000 * 3)
                self.assertAlmostEqual(hybrid_res["hits"][-1]["_score"], unmodified_rrf_scores["doc10"] * -1000 * 3)

    def test_hybrid_search_global_score_modifiers(self):
        """
        Tests that global score modifiers work as expected for RRF / Disjunction
        Make sure scores of modified results are calculated correctly based on unmodified scores.
        Ensures new result order reflects modified scores.
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

                unmodified_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="HELLO WORLD",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        verbose=True
                    ),
                    result_count=10
                )

                unmodified_scores = {}
                for hit in unmodified_res["hits"]:
                    unmodified_scores[hit["_id"]] = hit["_score"]

                modified_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="HELLO WORLD",
                    search_method="HYBRID",
                    score_modifiers=ScoreModifierLists(**{
                        "multiply_score_by": [
                            {"field_name": "mult_field_1", "weight": 1000},
                            {"field_name": "mult_field_2", "weight": -1000}
                        ],
                        "add_to_score": [
                            {"field_name": "add_field_1", "weight": 5}
                        ]
                    }),
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        verbose=True
                    ),
                    result_count=10
                )

                self.assertIn("hits", modified_res)

                self.assertEqual(modified_res["hits"][0]["_id"], "doc9")  # highest score (score*1000*3)
                self.assertAlmostEqual(modified_res["hits"][0]["_score"], unmodified_scores["doc9"] * 1000 * 3)

                self.assertEqual(modified_res["hits"][1]["_id"], "doc8")  # (score*1000*2)
                self.assertAlmostEqual(modified_res["hits"][1]["_score"], unmodified_scores["doc8"] * 1000 * 2)

                self.assertEqual(modified_res["hits"][2]["_id"], "doc7")  # (score + 5*1)
                self.assertAlmostEqual(modified_res["hits"][2]["_score"], unmodified_scores["doc7"] + 5*1)

                self.assertEqual(modified_res["hits"][3]["_id"], "doc6")  # (score)
                self.assertEqual(modified_res["hits"][3]["_score"], unmodified_scores["doc6"])

                self.assertEqual(modified_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-1000*3)
                self.assertAlmostEqual(modified_res["hits"][-1]["_score"], unmodified_scores["doc10"] * -1000 * 3)

                # Show that we can use just 1 or the other (multiply or add)
                with self.subTest("Only multiply_score_by"):
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="HELLO WORLD",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1000},
                                {"field_name": "mult_field_2", "weight": -1000}
                            ],
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=10
                    )

                    self.assertEqual(modified_res["hits"][0]["_id"], "doc9")
                    self.assertAlmostEqual(modified_res["hits"][0]["_score"], unmodified_scores["doc9"] * 1000 * 3)

                    self.assertEqual(modified_res["hits"][0]["_id"], "doc9")  # highest score (score*1000*3)
                    self.assertAlmostEqual(modified_res["hits"][0]["_score"], unmodified_scores["doc9"] * 1000 * 3)

                    self.assertEqual(modified_res["hits"][1]["_id"], "doc8")  # (score*1000*2)
                    self.assertAlmostEqual(modified_res["hits"][1]["_score"], unmodified_scores["doc8"] * 1000 * 2)

                    # doc6 and doc7 have the same score, so their order is non-deterministic
                    self.assertSetEqual({'doc6', 'doc7'}, {hit["_id"] for hit in modified_res["hits"][2:4]})
                    for hits in modified_res["hits"][2:4]:
                        self.assertEqual(hits["_score"], unmodified_scores[hits["_id"]])

                    self.assertEqual(modified_res["hits"][-1]["_id"], "doc10")  # lowest score (score*-1000*3)
                    self.assertAlmostEqual(modified_res["hits"][-1]["_score"], unmodified_scores["doc10"] * -1000 * 3)

                with self.subTest("Only add_to_score"):
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="HELLO WORLD",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 5}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=10
                    )

                    self.assertEqual(modified_res["hits"][0]["_id"], "doc7")
                    self.assertAlmostEqual(modified_res["hits"][0]["_score"], unmodified_scores["doc7"] + 5*1)
                    for hits in modified_res["hits"][1:]:
                        self.assertEqual(hits["_score"], unmodified_scores[hits["_id"]])


    def test_hybrid_search_global_score_modifiers_with_rerank_depth(self):
        """
        Tests that global score modifiers work as expected for RRF / Disjunction with rerankDepth
        Make sure scores of modified results are calculated correctly based on unmodified scores.
        Return 'limit' results whenever possible. If rerankDepth < limit, add on the extra unranked results after reranking.
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "tensor1", "text_tensor_only": "dog", "add_field_1": 1.0, "mult_field_1": 1.0},    # tensor HIGH score, no lexical match
                            {"_id": "tensor2", "text_tensor_only": "something completely unrelated. garbage.", "add_field_1": 2.0, "mult_field_1": 2.0},    # tensor LOW score, no lexical match
                            {"_id": "lexical1", "text_lexical_only": "dogs dogs", "add_field_1": -2.0, "mult_field_1": -2.0},    # lexical HIGH score, no tensor
                            {"_id": "lexical2", "text_lexical_only": "dogs", "add_field_1": -1.0, "mult_field_1": -1.0},  # lexical LOWER score, no tensor
                            {"_id": "both1", "text_field_1": "dogs dogs", "add_field_1": 0.0001}     # both tensor and lexical, HIGHEST rank in each list.
                        ],
                        tensor_fields=["text_tensor_only", "text_field_1"] \
                            if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                unmodified_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        verbose=True
                    ),
                    result_count=5
                )

                unmodified_scores = {}
                for hit in unmodified_res["hits"]:
                    unmodified_scores[hit["_id"]] = hit["_score"]

                # Order with score modifiers: tensor2, tensor1, both1, lexical2, lexical1
                # Order without score modifiers: both1, (tensor1 or lexical1), (tensor1 or lexical1), (tensor2 or lexical2), (tensor2 or lexical2)
                self.assertEqual(len(unmodified_res["hits"]), 5)
                self.assertEqual("both1", unmodified_res["hits"][0]["_id"])
                self.assertEqual(set([hit["_id"] for hit in unmodified_res["hits"][1:3]]), {"tensor1", "lexical1"})
                self.assertEqual(set([hit["_id"] for hit in unmodified_res["hits"][3:5]]), {"tensor2", "lexical2"})

                with self.subTest(f"Case 1: limit == rerankDepth == result.size()"):
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=5,
                        rerank_depth=5
                    )
                    # Order with score modifiers: tensor2, tensor1, both1, lexical2, lexical1
                    self.assertEqual(["tensor2", "tensor1", "both1", "lexical2", "lexical1"],
                                     [hit["_id"] for hit in modified_res["hits"]])
                    # Assert scores are all correctly modified
                    self.assertAlmostEqual(modified_res["hits"][0]["_score"], 2*unmodified_scores["tensor2"] + 2)
                    self.assertAlmostEqual(modified_res["hits"][1]["_score"], 1*unmodified_scores["tensor1"] + 1)
                    self.assertAlmostEqual(modified_res["hits"][2]["_score"], unmodified_scores["both1"] + 0.0001)
                    self.assertAlmostEqual(modified_res["hits"][3]["_score"], -1*unmodified_scores["lexical2"] - 1)
                    self.assertAlmostEqual(modified_res["hits"][4]["_score"], -2*unmodified_scores["lexical1"] - 2)

                with self.subTest(f"Case 2: limit < rerankDepth < hits.size()"):
                    # Rerank the top 3, then take the top 2 from there.
                    # Original top 3: both1, lexical1, tensor1. Rerank --> tensor1, both1, lexical1
                    # Top 2 after: tensor1, both1
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=2,
                        rerank_depth=3
                    )
                    self.assertEqual(len(modified_res["hits"]), 2)
                    self.assertEqual(["tensor1", "both1"], [hit["_id"] for hit in modified_res["hits"]])

                with self.subTest(f"Case 3: limit == rerankDepth < hits.size()"):
                    # Rerank the top 3, then only take the top 3.
                    # Original top 2: both1, (lexical1 or tensor1). Rerank --> tensor1, both1, lexical1
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=3,
                        rerank_depth=3
                    )
                    self.assertEqual(len(modified_res["hits"]), 3)
                    self.assertEqual(["tensor1", "both1", "lexical1"], [hit["_id"] for hit in modified_res["hits"]])

                with self.subTest(f"Case 4: limit < hits.size() < rerankDepth"):
                    # Attempt to rerank top 10 (will only be able to do 5), then return 4.
                    # Even though rerankDepth > 2*limit, we just rerank the highest number of results possible
                    # Original top 5: both1, (tensor1 or lexical1), (tensor1 or lexical1), (tensor2 or lexical2), (tensor2 or lexical2)
                    # Reranked top 5: tensor2, tensor1, both1, lexical2, lexical1
                    # Top 4 after: tensor2, tensor1, both1, lexical2
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=4,
                        rerank_depth=10
                    )
                    self.assertEqual(len(modified_res["hits"]), 4)
                    self.assertEqual(["tensor2", "tensor1", "both1", "lexical2"], [hit["_id"] for hit in modified_res["hits"]])

                with self.subTest(f"Case 5: rerankDepth < hits.size() < limit"):
                    # Rerank the top 3, but attempt to return 10 (will only be able to do 5).
                    # We can't control whether tensor1 or lexical1 (same rank in their respective lists) is first,
                    #   since it's sorted alphabetically by randomized Vespa ID
                    # Original top 3: both1, (tensor1 or lexical1), (tensor1 or lexical1)
                    # Reranked top 3: tensor1, both1, lexical1
                    # Top 5 after (remaining unranked 2 hits are added with original score):
                    #       [tensor1, both1, tensor2, lexical2, lexical1] or
                    #       [tensor1, both1, lexical2, tensor2, lexical1]
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=10,
                        rerank_depth=3
                    )
                    self.assertEqual(len(modified_res["hits"]), 5)

                    possible_results = [
                        ["tensor1", "both1", "tensor2", "lexical2", "lexical1"],
                        ["tensor1", "both1", "lexical2", "tensor2", "lexical1"]
                    ]
                    self.assertIn([hit["_id"] for hit in modified_res["hits"]], possible_results)

                    # Check that last 2 hits do NOT have scores modified while the first 3 do
                    for hit in modified_res["hits"]:
                        if hit["_id"] == "tensor1":
                            self.assertEqual(hit["_score"], 1*unmodified_scores[hit["_id"]] + 1)
                        elif hit["_id"] == "both1":
                            self.assertAlmostEqual(hit["_score"], unmodified_scores[hit["_id"]] + 0.0001)
                        elif hit["_id"] in ["tensor2", "lexical2"]:
                            self.assertEqual(hit["_score"], unmodified_scores[hit["_id"]])
                        elif hit["_id"] == "lexical1":
                            self.assertEqual(hit["_score"], -2*unmodified_scores[hit["_id"]] - 2)

                with self.subTest("Case 6: 2*limit < rerankDepth"):
                    # We attempt to rerank more hits than what is possible to retrieve (tensor + lexical search)
                    # Initial search will give us both1 in both tensor and lexical
                    # Only 1 hit to rerank
                    # Trim to just both1
                    modified_res = tensor_search.search(
                        config=self.config,
                        index_name=index.name,
                        text="dogs",
                        search_method="HYBRID",
                        score_modifiers=ScoreModifierLists(**{
                            "multiply_score_by": [
                                {"field_name": "mult_field_1", "weight": 1},
                            ],
                            "add_to_score": [
                                {"field_name": "add_field_1", "weight": 1}
                            ]
                        }),
                        hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF,
                            verbose=True
                        ),
                        result_count=1,
                        rerank_depth=4
                    )
                    self.assertEqual(len(modified_res["hits"]), 1)
                    self.assertEqual("both1", modified_res["hits"][0]["_id"])
                    # Ensure score is modified
                    self.assertAlmostEqual(modified_res["hits"][0]["_score"], unmodified_scores["both1"] + 0.0001)

            with self.subTest("Case 7: rerankDepth == 0"):
                # Result order and scores should be same as original search
                modified_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    score_modifiers=ScoreModifierLists(**{
                        "multiply_score_by": [
                            {"field_name": "mult_field_1", "weight": 1},
                        ],
                        "add_to_score": [
                            {"field_name": "add_field_1", "weight": 1}
                        ]
                    }),
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        verbose=True
                    ),
                    result_count=5,
                    rerank_depth=0
                )
                self.assertEqual(len(modified_res["hits"]), 5)
                self.assertEqual(len(unmodified_res["hits"]), 5)
                self.assertEqual("both1", unmodified_res["hits"][0]["_id"])
                self.assertEqual(set([hit["_id"] for hit in unmodified_res["hits"][1:3]]), {"tensor1", "lexical1"})
                self.assertEqual(set([hit["_id"] for hit in unmodified_res["hits"][3:5]]), {"tensor2", "lexical2"})

                for hit in modified_res["hits"]:
                    self.assertEqual(hit["_score"], unmodified_scores[hit["_id"]])

            with self.subTest("Case 8: No rerankDepth"):
                # Set limit to 3 so all results are included, but since no rerankDepth, it will rerank everything
                # Original top all: both1, (tensor1 or lexical1), (tensor1 or lexical1), (tensor2 or lexical2), (tensor2 or lexical2)
                # Reranked top all: tensor2, tensor1, both1, lexical2, lexical1
                # Top 3 after: tensor2, tensor1, both1
                modified_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="HYBRID",
                    score_modifiers=ScoreModifierLists(**{
                        "multiply_score_by": [
                            {"field_name": "mult_field_1", "weight": 1},
                        ],
                        "add_to_score": [
                            {"field_name": "add_field_1", "weight": 1}
                        ]
                    }),
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        verbose=True
                    ),
                    result_count=3
                )
                self.assertEqual(len(modified_res["hits"]), 3)
                self.assertEqual(["tensor2", "tensor1", "both1"], [hit["_id"] for hit in modified_res["hits"]])


    @pytest.mark.skip_for_multinode
    def test_hybrid_search_lexical_tensor_with_lexical_score_modifiers_succeeds(self):
        """
        Tests that if we do hybrid search with lexical retrieval and tensor ranking, we can use both lexical and tensor
        score modifiers.

        The lexical score modifiers should affect the actual result set, while the tensor score modifiers should
        affect the order and score.
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

    @pytest.mark.skip_for_multinode
    def test_hybrid_search_same_retrieval_and_ranking_matches_original_method(self):
        """
        Tests that hybrid search with:
        retrieval_method = "lexical", ranking_method = "lexical" and
        retrieval_method = "tensor", ranking_method = "tensor"

        Results must be the same as lexical search and tensor search respectively.
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
                # Add documents
                self.add_documents(
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

    @pytest.mark.skip_for_multinode
    def test_hybrid_search_with_images(self):
        """
        Tests that hybrid search is accurate with images, both in query and in documents.
        Note: For unstructured, both image and link are indexed, thus the doc will have a lexical and tensor score.
        For structured, only the image is indexed, thus the doc will ONLY have a tensor score.
        """

        for index in [self.structured_default_image_index, self.semi_structured_default_image_index,
                      self.unstructured_default_image_index]:
            with self.subTest(index=index.name):
                # Add documents
                self.add_documents(
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
        self.add_documents(
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

    def test_hybrid_search_semi_structured_opposite_retrieval_and_ranking(self):
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
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_default_image_index.name,
                docs=self.docs_list,
                tensor_fields=["text_field_1", "text_field_2"]
            )
        )

        # Reference results
        tensor_res_all_docs = tensor_search.search(  # To get tensor scores of every doc, for reference
            config=self.config,
            index_name=self.semi_structured_default_image_index.name,
            text="dogs",
            search_method="TENSOR",
            result_count=20,
            searchable_attributes=["text_field_1"]
        )
        lexical_res = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_default_image_index.name,
            text="dogs",
            search_method="LEXICAL",
            result_count=10,
            searchable_attributes=["text_field_1"]
        )
        tensor_res = tensor_search.search(
            config=self.config,
            index_name=self.semi_structured_default_image_index.name,
            text="dogs",
            search_method="TENSOR",
            result_count=10,
            searchable_attributes=["text_field_1"]
        )

        # Lexical retrieval with Tensor ranking
        with self.subTest(retrievalMethod=RetrievalMethod.Lexical, rankingMethod=RankingMethod.Tensor):
            hybrid_res = tensor_search.search(
                config=self.config,
                index_name=self.semi_structured_default_image_index.name,
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
                index_name=self.semi_structured_default_image_index.name,
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

    def test_hybrid_search_highlights_for_lexical_tensor(self):
        """
        Tests that hybrid search with highlights:
        retrievalMethod = "lexical", rankingMethod = "tensor"

        has expected results and highlights even on results that have a non-tensor field.
        No highlights on the results retrieved from the non-tensor field (text_field_2 in this case),
        so list should be empty.
        """
        for index in [self.unstructured_default_image_index, self.semi_structured_default_image_index,
                      self.structured_index_one_tensor_field]:
            with self.subTest(msg=f'{index.type}', index=index):

                # Add documents
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Reference results
                tensor_res_all_docs = tensor_search.search(  # To get tensor scores of every doc, for reference
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="TENSOR",
                    result_count=20,
                )
                lexical_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="dogs",
                    search_method="LEXICAL",
                    result_count=10,
                )

                # Lexical retrieval with Tensor ranking
                hybrid_res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
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

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=index.name):
                if isinstance(index, (StructuredMarqoIndex, SemiStructuredMarqoIndex)):
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

    def test_hybrid_search_searchable_attributes_fails(self):
        """
        Ensure that searchable_attributes cannot be set in hybrid search.
        """

        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index,
                      self.unstructured_default_text_index]:
            with self.subTest(index=type(index)):
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

    def test_hybrid_search_score_modifiers_old_version_fails(self):
        """
        score_modifiers can only be set for hybrid Marqo 2.15.0 onward
        """
        # Legacy Index too old for root score_modifiers
        for index in [self.unstructured_default_text_index, self.semi_structured_text_index_2_14,
                      self.structured_text_index_2_14]:
            with self.subTest(index=type(index)):
                with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
                    tensor_search.search(
                        config=self.config,
                        index_name=self.unstructured_default_text_index.name,
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
                self.assertIn("global score modifiers is only supported for "
                              "Marqo indexes created with Marqo 2.15.0", str(e.exception))

    def test_hybrid_search_rerank_depth_old_version_fails(self):
        """
        rerank_depth can only be set for hybrid Marqo 2.15.0 onward
        """
        # Legacy Index too old for root score_modifiers
        for index in [self.unstructured_default_text_index, self.semi_structured_text_index_2_14,
                      self.structured_text_index_2_14]:
            with self.subTest(index=type(index)):
                with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
                    res = tensor_search.search(
                        config=self.config,
                        index_name=self.unstructured_default_text_index.name,
                        text="dogs",
                        search_method="HYBRID",
                        rerank_depth=5
                    )
                self.assertIn("'rerankDepth' search parameter is only supported for indexes created "
                              "with Marqo version 2.15.0", str(e.exception))

    def test_hybrid_search_score_modifiers_wrong_ranking_method_fails(self):
        # Structured / semi-structured score modifiers but not RRF
        with self.subTest("score_modifiers for structured/semi-structured but not RRF ranking"):
            for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
                with self.subTest(index=type(index)):
                    for retrieval_method, ranking_method in [
                        (RetrievalMethod.Tensor, RankingMethod.Lexical),
                        (RetrievalMethod.Lexical, RankingMethod.Tensor),
                        (RetrievalMethod.Tensor, RankingMethod.Tensor),
                        (RetrievalMethod.Lexical, RankingMethod.Lexical)
                    ]:
                        with self.assertRaises(ValueError) as e:
                            tensor_search.search(
                                config=self.config,
                                index_name=index.name,
                                text="dogs",
                                search_method="HYBRID",
                                hybrid_parameters=HybridParameters(
                                    retrievalMethod=retrieval_method,
                                    rankingMethod=ranking_method
                                ),
                                score_modifiers=ScoreModifierLists(
                                    multiply_score_by=[
                                        {"field_name": "mult_field_1", "weight": 1.0}
                                    ],
                                    add_to_score=[
                                        {"field_name": "add_field_1", "weight": 1.0}
                                    ]
                                ),
                            )
                        self.assertIn("if 'rankingMethod' is 'RRF'", str(e.exception))

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
                _ = self.add_documents(config=self.config,
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
        Test that hybrid search with legacy unstructured index and searchable attributes fails.
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

    def test_hybrid_with_two_errors_returns_both(self):
        """
        If vespa query to the hybrid searcher returns a result with 2 errors, both should be in the error message.
        If all errors are timeout, raise VespaTimeoutError (504).
        If even one error is not timeout, raise VespaStatusError (500).
        """

        # Mock Vespa result with 2 errors
        test_cases = [
            # HTTP 504, first is Vespa 12
            (
                {
                    'root': {
                        'relevance': 1.0,
                        'fields': {'totalCount': 0},
                        'errors': [
                            {
                                'code': 12,
                                'summary': 'Timed out',
                                'source': 'content_default',
                                'message': "Error in execution of chain 'content_default': Chain timed out."
                            },
                            {
                                'code': 4,
                                'summary': 'Invalid query parameter',
                                'message': 'Could not create query from YQL.'
                            }
                        ]
                    }
                },
                504,    # HTTP 504
                False,  # Not a timeout, since 2nd error is not timeout
            ),
            # HTTP 400, second is Vespa 12
            (
                {
                    'root': {
                        'relevance': 1.0,
                        'fields': {'totalCount': 0},
                        'errors': [
                            {
                                'code': 4,
                                'summary': 'Invalid query parameter',
                                'message': 'Could not create query from YQL.'
                            },
                            {
                                'code': 12,
                                'summary': 'Timed out',
                                'source': 'content_default',
                                'message': "Error in execution of chain 'content_default': Chain timed out."
                            }
                        ]
                    }
                },
                400,  # HTTP 400
                False,  # Not a timeout, since 1st error is not timeout
            ),
            # HTTP 504, both Vespa errors 12
            (
                {
                    'root': {
                        'relevance': 1.0,
                        'fields': {'totalCount': 0},
                        'errors': [
                            {
                                'code': 12,
                                'summary': 'Timed out',
                                'source': 'content_default',
                                'message': "Error in execution of chain 'content_default': Chain timed out."
                            },
                            {
                                'code': 12,
                                'summary': 'Timed out',
                                'source': 'content_default',
                                'message': "Error in execution of chain 'content_default': Chain timed out."
                            }
                        ]
                    }
                },
                504,    # HTTP 504
                True    # Timeout, since both errors are timeout
            ),
            # HTTP 400, both Vespa errors 12 but not timeout
            (
                {
                    'root': {
                        'relevance': 1.0,
                        'fields': {'totalCount': 0},
                        'errors': [
                            {
                                'code': 12,
                                'summary': 'Some other error',
                                'source': 'content_default',
                                'message': 'Some error message'
                            },
                            {
                                'code': 12,
                                'summary': 'Some other error',
                                'source': 'content_default',
                                'message': 'Some error message'
                            }
                        ]
                    }
                },
                400,    # HTTP 400
                False    # Not a timeout, since not 504 error code
            ),
            # HTTP 504, first 12, second soft doom
            (
                {
                    'root': {
                        'relevance': 1.0,
                        'fields': {'totalCount': 0},
                        'errors': [
                            {
                                'code': 12,
                                'summary': 'Timed out',
                                'source': 'content_default',
                                'message': "Error in execution of chain 'content_default': Chain timed out."
                            },
                            {
                                'code': 8,
                                'summary': 'Soft doom',
                                'message': 'Search request soft doomed during query setup and initialization.'
                            }
                        ]
                    }
                },
                504,    # HTTP 504
                True    # Timeout, since both errors are timeout
            )
        ]

        for result_dict, vespa_httpx_code, should_be_timeout in test_cases:
            with self.subTest(result_dict=result_dict, vespa_httpx_code=vespa_httpx_code,
                              should_be_timeout=should_be_timeout):
                # If should_be_timeout, raise a VespaTimeoutError (504), else raise a VespaStatusError (500)
                mock_vespa_result = httpx.Response(
                    status_code=vespa_httpx_code,
                    content=json.dumps(result_dict),
                    request=httpx.Request("POST", "http://localhost:8080/test-url/")
                )

                with unittest.mock.patch("httpx.Client.post") as mock_query:
                    mock_query.return_value = mock_vespa_result

                    for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
                        with self.subTest(index=type(index)):
                            with self.assertRaises(vespa_exceptions.VespaStatusError) as e:
                                tensor_search.search(
                                    text='dogs', config=self.config, index_name=index.name,
                                    search_method=SearchMethod.HYBRID
                                )
                            self.assertEqual(should_be_timeout,
                                             isinstance(e.exception, vespa_exceptions.VespaTimeoutError))

                            for error in result_dict["root"]["errors"]:
                                # All error messages should be in final exception
                                self.assertIn(error["message"], str(e.exception))
                                self.assertIn(error["summary"], str(e.exception))

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

    def test_correct_error_is_raised_if_rerank_depth_is_provided_on_old_marqo_version(self):
        """
        Tests that an error is raised if rerank_depth is provided on an old marqo version.
        """

        with self.assertRaises(core_exceptions.UnsupportedFeatureError) as e:
            tensor_search.search(
                config=self.config,
                index_name=self.unstructured_default_text_index.name,
                text="dogs",
                search_method="HYBRID",
                rerank_depth=3,
                result_count=3
            )

        self.assertIn("Marqo version 2.15.0 or later", str(e.exception))

        _ = tensor_search.search(
            config=self.config,
            index_name=self.unstructured_default_text_index.name,
            text="dogs",
            search_method="HYBRID",
            # rerank_depth=3, # This should not raise an error
            result_count=3
        )

    def test_rerank_depth_tensor_hybrid_search(self):
        """Test hybrid search with rerankDepthTensor across different scenarios."""

        docs = [{
            "_id": f"doc_{i}",
            "text_field_1": f"sample text {i}"
        } for i in range(10)]

        for index in [self.semi_structured_default_text_index, self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                tensor_fields = ["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None

                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=docs, tensor_fields=tensor_fields
                    )
                )

                base_kwargs = dict(
                    config=self.config, index_name=index.name, text="sample text", search_method="HYBRID"
                )

                # Case 1: rerankDepthTensor < result_count → result_count is respected
                with self.subTest(case="rerankDepthTensor_limits_final_hits"):
                    res = tensor_search.search(
                        **base_kwargs, result_count=5, hybrid_parameters=HybridParameters(
                            rerankDepthTensor=3, verbose=True, retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        )
                    )
                    self.assertEqual(len(res["hits"]), 5)

                # Case 2: rerankDepthTensor > result_count + offset → return full page
                with self.subTest(case="rerank_depth_greater_than_offset_plus_limit"):
                    res = tensor_search.search(
                        **base_kwargs, result_count=3, offset=2, hybrid_parameters=HybridParameters(
                            rerankDepthTensor=10, verbose=True, retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        )
                    )
                    self.assertEqual(len(res["hits"]), 3)

                # Case 3: rerankDepthTensor < offset → limit + offset are respected
                with self.subTest(case="offset_beyond_rerank_depth"):
                    res = tensor_search.search(
                        **base_kwargs, result_count=1, offset=5, hybrid_parameters=HybridParameters(
                            rerankDepthTensor=3, verbose=True, retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        )
                    )
                    self.assertGreaterEqual(len(res["hits"]), 1)

                # Case 4: rerankDepthTensor omitted → return full limit
                with self.subTest(case="no_rerankDepthTensor"):
                    res = tensor_search.search(
                        **base_kwargs, result_count=10, hybrid_parameters=HybridParameters(
                            retrievalMethod=RetrievalMethod.Tensor, rankingMethod=RankingMethod.Tensor
                        )
                    )
                    self.assertEqual(len(res["hits"]), 10)

    def test_weighted_tensor_query(self):
        """
        Tests that a weighted tensor query can be made.
        """

        # Add documents
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                # Adding documents
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Reference results
                tensor_queries_list = [
                    # tensor query with 1 field
                    [{"dogs": 1.0}, {"dogs": -1}],
                    # tensor query with 3 fields
                    [{"dogs": 1.0, "cats": 0.5, "birds": 0.25}, {"dogs": -1, "cats": -0.5, "birds": -0.25}],
                ]
                for tensor_query in tensor_queries_list:
                    with self.subTest(tensor_query=tensor_query):
                        # Results with normal weights
                        tensor_res = tensor_search.search(
                            config=self.config,
                            index_name=self.structured_text_index_score_modifiers.name,
                            search_method="HYBRID",
                            text=None,
                            hybrid_parameters=HybridParameters(
                                queryTensor=tensor_query[0],
                                retrievalMethod=RetrievalMethod.Tensor,
                                rankingMethod=RankingMethod.Tensor
                            ),
                            result_count=10
                            )

                        # Results with reverse weights
                        tensor_res_reverse = tensor_search.search(
                            config=self.config,
                            index_name=self.structured_text_index_score_modifiers.name,
                            search_method="HYBRID",
                            text=None,
                            hybrid_parameters=HybridParameters(
                                queryTensor=tensor_query[1],
                                retrievalMethod=RetrievalMethod.Tensor,
                                rankingMethod=RankingMethod.Tensor
                            ),
                            result_count=5
                        )

                        # Check that top result is not present in reverse weighted query
                        top_hit = tensor_res["hits"][0]
                        self.assertIsNone(
                            next((doc for doc in tensor_res_reverse["hits"] if doc["_id"] == top_hit["_id"]), None)
                        )

    def test_different_retrieval_and_ranking_combinations_with_weighted_queries(self):
        """
        Tests that different search and retrieval combinations can be made with weighted queries.
        """

        # Add documents
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                # Adding documents
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                base_parameters = {
                    "config": self.config,
                    "index_name": index.name,
                    "search_method": "HYBRID",
                    "text": None,
                }

                with self.subTest("Lexical retrieval, Tensor ranking - opposite weights should provide opposite results"):
                    res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": 1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Tensor
                        ),
                        result_count=10
                    )

                    reverse_res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": -1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Tensor
                        ),
                        result_count=5
                    )

                    # Check that top result is lowest in reverse weighted query
                    assert res["hits"][0]["_id"] == reverse_res["hits"][-1]["_id"]

                with self.subTest("tensor retrieval, lexical ranking - opposite weights result should not be included"):
                    res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": 1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Lexical
                        ),
                        result_count=10
                    )

                    reverse_res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": -1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Lexical
                        ),
                        result_count=5
                    )

                    # Check that top result is lowest in reverse weighted query
                    assert not any(r["_id"] for r in reverse_res["hits"] if r["_id"] == res["hits"][0]["_id"])

                with self.subTest("tensor retrieval, tensor ranking - opposite weights result should not be included"):
                    res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": 1.0,
                            },
                            queryLexical=None,
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        ),
                        result_count=10
                    )

                    reverse_res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": -1.0,
                            },
                            queryLexical=None,
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        ),
                        result_count=5
                    )

                    # Check that top result is lowest in reverse weighted query
                    assert not any(r["_id"] for r in reverse_res["hits"] if r["_id"] == res["hits"][0]["_id"])

                with self.subTest("disjunction retrieval, RRF ranking - opposite weights should provide opposite results"):
                    res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": 1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ),
                        result_count=10
                    )

                    reverse_res = tensor_search.search(
                        **base_parameters,
                        hybrid_parameters=HybridParameters(
                            queryTensor={
                                "dogs": -1.0,
                            },
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ),
                        result_count=5
                    )

                    # Check that top result is lowest in reverse weighted query
                    assert res["hits"][0]["_id"] == reverse_res["hits"][-1]["_id"]



    def test_lexical_retrieval_tensor_rerank_with_weighted_query(self):
        """
        Tests that a weighted tensor query can be made.
        """

        # Add documents
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                # Adding documents
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Reference results
                res = tensor_search.search(
                    config=self.config, index_name=self.structured_text_index_score_modifiers.name,
                    search_method="HYBRID", text=None, hybrid_parameters=HybridParameters(
                        queryTensor={
                            "dogs": 1.0,
                        },
                        queryLexical="dogs",
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Tensor
                    ), result_count=5
                )

                # Results with reverse weights for dogs
                res_reverse = tensor_search.search(
                    config=self.config, index_name=self.structured_text_index_score_modifiers.name,
                    search_method="HYBRID", text=None, hybrid_parameters=HybridParameters(
                        queryTensor={
                            "dogs": -1.0,
                        },
                        queryLexical="dogs",
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Tensor
                    ), result_count=5
                )

                # Check that top result is lowest in reverse weighted query
                assert res["hits"][0]["_id"] == res_reverse["hits"][-1]["_id"]

    def test_empty_tensor_query_dict(self):
        """Ensure empty tensor query dict does not raise errors and behaves correctly."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(ValueError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, text=None, search_method="HYBRID",
                        hybrid_parameters=HybridParameters(
                            queryTensor={},  # Edge case
                            queryLexical="dogs", retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ), result_count=5
                    )

    def test_query_tensor_as_string_equivalent_to_single_query(self):
        """String tensor query should work like dict with one key."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                string_query = "dogs"
                dict_query = {
                    "dogs": 1.0
                }

                str_res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor=string_query,
                        rankingMethod=RankingMethod.Tensor,
                        retrievalMethod=RetrievalMethod.Tensor
                    ), result_count=5
                )

                dict_res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor=dict_query,
                        rankingMethod=RankingMethod.Tensor,
                        retrievalMethod=RetrievalMethod.Tensor
                    ), result_count=5
                )

                self.assertEqual(
                    [hit["_id"] for hit in str_res["hits"]], [hit["_id"] for hit in dict_res["hits"]],
                    "String queryTensor should behave like single-entry dict"
                )

    def test_query_tensor_as_multi_values_dict_produces_different_results_to_single_value_dict(self):
        """Ensure that a query tensor with multiple values produces different results to a single value dict."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                single_query = {
                    "dogs": 1.0
                }

                multi_query = {
                    "dogs": 1.0,
                    "cats": 0.5
                }

                single_res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor=single_query,
                        retrievalMethod=RetrievalMethod.Tensor,
                        rankingMethod=RankingMethod.Tensor
                    ), result_count=5
                )

                multi_res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor=multi_query,
                        retrievalMethod=RetrievalMethod.Tensor,
                        rankingMethod=RankingMethod.Tensor
                    ), result_count=5
                )

                for res in single_res['hits']:
                    res_with_same_id = next((r for r in multi_res['hits'] if r['_id'] == res['_id']), None)
                    if res_with_same_id:
                        self.assertNotEqual(res['_score'], res_with_same_id['_score'])

    def test_none_query_tensor(self):
        """Ensure that a None query tensor raises an errors."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor=None,
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor
                        ), result_count=5
                    )

    def test_none_query_lexical(self):
        """Ensure that a None query lexical raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryLexical=None,
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Lexical
                        ), result_count=5
                    )

    def test_none_query_lexical_and_tensor_disjunction_retrieval(self):
        """Ensure that a None query lexical and tensor with disjunction retrieval raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor=None,
                            queryLexical=None,
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ), result_count=5
                    )

    def test_none_query_lexical_disjunction_retrieval(self):
        """Ensure that a None query lexical and tensor with disjunction retrieval raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor="dogs",
                            queryLexical=None,
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ), result_count=5
                    )

    def test_none_query_tensor_disjunction_retrieval(self):
        """Ensure that a None query tensor and lexical with disjunction retrieval raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor=None,
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Disjunction,
                            rankingMethod=RankingMethod.RRF
                        ), result_count=5
                    )

    def test_none_provided_for_tensor_lexical_retrieval_works(self):
        """Ensure that None can be provided for tensor query when retrievalMethod and rankingMethod are lexical."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor=None,
                        queryLexical="dogs",
                        retrievalMethod=RetrievalMethod.Lexical,
                        rankingMethod=RankingMethod.Lexical,
                    ), result_count=5
                )

                self.assertIn("hits", res)

    def test_none_provided_for_lexical_tensor_retrieval_works(self):
        """Ensure that None can be provided for lexical query when retrievalMethod and rankingMethod are Tensor."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                res = tensor_search.search(
                    config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                    hybrid_parameters=HybridParameters(
                        queryTensor="dogs",
                        queryLexical=None,
                        retrievalMethod=RetrievalMethod.Tensor,
                        rankingMethod=RankingMethod.Tensor,
                    ), result_count=5
                )

                self.assertIn("hits", res)

    def test_tensor_query_provided_for_lexical_retrieval_lexical_ranking_raises_error(self):
        """Ensure that providing a tensor query for lexical retrieval and ranking raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor="dogs",
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Lexical,
                            rankingMethod=RankingMethod.Lexical,
                        ), result_count=5
                    )

    def test_lexical_query_provided_for_tensor_retrieval_tensor_ranking_raises_error(self):
        """Ensure that providing a lexical query for tensor retrieval and ranking raises an error."""
        for index in [self.structured_text_index_score_modifiers, self.semi_structured_default_text_index]:
            with self.subTest(index=index.type):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=index.name, docs=self.docs_list,
                        tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                with self.assertRaises(InvalidArgumentError):
                    tensor_search.search(
                        config=self.config, index_name=index.name, search_method="HYBRID", text=None,
                        hybrid_parameters=HybridParameters(
                            queryTensor=None,
                            queryLexical="dogs",
                            retrievalMethod=RetrievalMethod.Tensor,
                            rankingMethod=RankingMethod.Tensor,
                        ), result_count=5
                    )
