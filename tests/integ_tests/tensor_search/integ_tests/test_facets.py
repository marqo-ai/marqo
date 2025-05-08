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


class TestFacets(MarqoTestCase):
    """
    Combined tests for unstructured and structured hybrid search.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.indexes = cls.create_indexes([
            semi_structured_default_text_index,
        ])

        cls.semi_structured_default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        super().setUp()

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def assert_dict_almost_equal(self, d1, d2, places=1):
        """Compare two dictionaries with numeric values allowing for small differences."""
        self.assertEqual(d1.keys(), d2.keys(), "Dictionaries have different keys")
        for key in d1:
            if isinstance(d1[key], dict):
                self.assert_dict_almost_equal(d1[key], d2[key], places)
            elif isinstance(d1[key], (int, float)) and isinstance(d2[key], (int, float)):
                self.assertAlmostEqual(d1[key], d2[key], places=places,
                                       msg=f"Values differ for key {key}: {d1[key]} != {d2[key]}")
            else:
                self.assertEqual(d1[key], d2[key],
                                 f"Values differ for key {key}: {d1[key]} != {d2[key]}")

    def _test_facet_query(self, retrieval_method, ranking_method, facets, expected_lexical_facets, expected_other_facets):
        res = tensor_search.search(
            config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
            facets=facets,
            search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                retrievalMethod=retrieval_method, rankingMethod=ranking_method
            )
        )
        if retrieval_method == RetrievalMethod.Lexical:
            # lexical only matches 1 doc for q="shirt"
            self.assert_dict_almost_equal(res["facets"], expected_lexical_facets)
        else:
            self.assert_dict_almost_equal(res["facets"], expected_other_facets)

    def add_fashion_docs(self, modified_docs=None):
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.semi_structured_default_text_index.name,
                docs=modified_docs if modified_docs is not None else EXAMPLE_FASHION_DOCUMENTS,
                tensor_fields=["title", "description"]
            )
        )

    def test_single_facet(self):
        """
        Tests facet requests for a single field (color) across different retrieval and ranking methods.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(type="string")})
                expected_lexical_facets = {'color': {'red': {'count': 1}}}
                expected_other_facets = {'color': {'red': {'count': 2}, 'coral': {'count': 1}, 'green': {'count': 1}}}
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )

    def test_single_facet_max_results(self):
        """
        Tests facet requests with a global maxResults limit that applies to all facet fields.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(type="string")}, maxResults=1)
                expected_lexical_facets = {'color': {'red': {'count': 1}}}
                expected_other_facets = {'color': {'red': {'count': 2}}}
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )
    def test_single_facet_max_results_override(self):
        """
        Tests that field-level maxResults override can exceed the global maxResults limit.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(type="string", maxResults=2)}, maxResults=1)
                expected_lexical_facets = {'color': {'red': {'count': 1}}}
                expected_other_facets = {'color': {'red': {'count': 2}, 'coral': {'count': 1}}}
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )
    
    def test_single_facet_reverse_ordering_returns_lower_resulst(self):
        """
        Tests ascending order facet results return facets with lower counts first.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(type="string")}, maxResults=1, order="asc")
                expected_lexical_facets = {'color': {'red': {'count': 1}}}
                expected_other_facets = {'color': {'coral': {'count': 1}}}
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )
    
    def test_single_facet_number_field(self):
        """
        Tests facet aggregations (min, max, avg, sum, count) for numeric fields.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={"price": FieldFacetsConfiguration(type="number")})
                expected_lexical_facets = {'price': {'avg': 49.03, 'count': 1, 'max': 49.03, 'min': 49.03, 'sum': 49.03}}
                expected_other_facets = {'price': {'sum': 224.55, 'avg': 56.1375, 'min': 1.2, 'max': 92.99, 'count': 4}}
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )
    def test_single_facet_number_field_with_ranges(self):
        """
        Tests numeric facets with custom range buckets using from/to configurations.
        """
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                facets = FacetsParameters(fields={
                    "price": FieldFacetsConfiguration(type="number", ranges=[{"to": 50}, {"from": 50}]),
                })
                expected_lexical_facets = {'price': {
                    '-Inf:50.0': {'avg': 49.03,'count': 1,'max': 49.03,'min': 49.03, 'sum': 49.03}
                }
                }
                expected_other_facets = {'price': {
                    '-Inf:50.0': {'avg': 25.115,'count': 2,'max': 49.03,'min': 1.2, 'sum': 50.23},
                    '50.0:Inf': {'avg': 87.16, 'count': 2, 'max': 92.99, 'min': 81.33, 'sum': 174.32}}
                }
                self._test_facet_query(
                    retrieval_method=retrieval_method, ranking_method=ranking_method,
                    facets=facets,
                    expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets
                )

    def test_single_number_facets_int(self):
        """
        Tests numeric facets behavior with integer values, including range boundaries
        and count validations across different retrieval methods.
        """
        docs_for_test = copy.deepcopy(EXAMPLE_FASHION_DOCUMENTS)
        for doc_for_test in docs_for_test:
            doc_for_test["price"] = int(doc_for_test["price"])
        self.add_fashion_docs(docs_for_test)
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                with self.subTest("Test expected number of fields returned"):
                    facets = FacetsParameters(fields={
                        "price": FieldFacetsConfiguration(type="number"),
                    })
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        )
                    )
                    if retrieval_method == RetrievalMethod.Lexical:
                        self.assertEqual(res["facets"]["price"]["count"], 1)
                    else:
                        self.assertEqual(res["facets"]["price"]["count"], 4)
                with self.subTest("Test ranges include up to but not including 'to' value and return expected counts"):
                    facets = FacetsParameters(fields={
                        "price": FieldFacetsConfiguration(type="number", ranges=[{"to": 49, "name": "to49"}, {"from": 49, "name": "from49"}]),
                    })
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        )
                    )
                    if retrieval_method == RetrievalMethod.Lexical:
                        # expected match here costs exactly 49 and is included in from=49 but excluded when to=49
                        self.assertEqual(res["facets"]["price"]["from49"]["count"], 1)
                    else:
                        self.assertEqual(res["facets"]["price"]["to49"]["count"], 1)
                        self.assertEqual(res["facets"]["price"]["from49"]["count"], 3)
                with self.subTest("Test ranges support both 'from' and 'to' and return expected counts"):
                    facets = FacetsParameters(fields={
                        "price": FieldFacetsConfiguration(type="number", ranges=[{"from": 3, "to": 50}, {"from": 50, "name": "from50"}]),
                    })
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        )
                    )
                    if retrieval_method == RetrievalMethod.Lexical:
                        self.assertEqual(res["facets"]["price"]["3.0:50.0"]["count"], 1)
                    else:
                        self.assertEqual(res["facets"]["price"]["3.0:50.0"]["count"], 1)
                        self.assertEqual(res["facets"]["price"]["from50"]["count"], 2)

    def test_array_facets(self):
        """
        Tests array field faceting by comparing results between individual field facets
        and combined array facets containing the same information.
        """
        docs_for_test = copy.deepcopy(EXAMPLE_FASHION_DOCUMENTS)
        for doc_for_test in docs_for_test:
            doc_for_test["tags"] = [
                f"color:{doc_for_test['color']}",
                f"brand:{doc_for_test['brand']}",
                f"style:{doc_for_test['style']}",
            ]

        self.add_fashion_docs(docs_for_test)
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                # Test regular facets and facets from array are the same
                facet_requests = [
                    FacetsParameters(fields={
                        "color": FieldFacetsConfiguration(type="string"),
                        "brand": FieldFacetsConfiguration(type="string"),
                        "style": FieldFacetsConfiguration(type="string"),
                    }),
                    FacetsParameters(fields={"tags": FieldFacetsConfiguration(type="array")}),
                ]
                res_string_facets = tensor_search.search(
                    config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                    facets=facet_requests[0],
                    search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method, rankingMethod=ranking_method
                    )
                )
                res_array_facets = tensor_search.search(
                    config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                    facets=facet_requests[1],
                    search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method, rankingMethod=ranking_method
                    )
                )
                self.assertNotEqual(res_array_facets, {})
                for facet, value in res_array_facets["facets"]["tags"].items():
                    splitted_name = facet.split(":")
                    self.assertDictEqual(res_string_facets["facets"][splitted_name[0]][splitted_name[1]], value)


    def test_facets_filter_term_exclusions(self):
        """
        Tests facet behavior with filter exclusions, verifying that excludeTerms properly
        removes specified terms from facet results while maintaining proper counts.
        """

        def test_facet_query(retrieval_method, ranking_method, facets, filter_string, expected_lexical_facets, expected_other_facets):
            res = tensor_search.search(
                config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                facets=facets,
                search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                    retrievalMethod=retrieval_method, rankingMethod=ranking_method
                ),
                filter=filter_string,
            )
            if retrieval_method == RetrievalMethod.Lexical:
                # lexical only matches 1 doc for q="shirt"
                self.assertDictEqual(res["facets"], expected_lexical_facets)
            else:
                self.assertDictEqual(res["facets"], expected_other_facets)
            return res

        self.add_fashion_docs(EXAMPLE_FASHION_DOCUMENTS)
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                with self.subTest("No facets returned due to filtering"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string"
                    )})
                    expected_lexical_facets = {}
                    expected_other_facets = {}
                    res = test_facet_query(
                        retrieval_method=retrieval_method, ranking_method=ranking_method,
                        facets=facets,
                        expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets,
                        filter_string="price:[100 TO 200]"
                    )
                    self.assertEqual(res["hits"], [])
                with self.subTest("No hits returned due to filtering, facets with terms exclusions are present"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string", excludeTerms=["price:[100 TO 200]"]
                    )})
                    expected_lexical_facets = {'color': {'red': {'count': 1}}}
                    expected_other_facets = {'color': {'red': {'count': 2}, 'coral': {'count': 1}, 'green': {'count': 1}}}
                    res = test_facet_query(
                        retrieval_method=retrieval_method, ranking_method=ranking_method,
                        facets=facets,
                        expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets,
                        filter_string="price:[100 TO 200]"
                    )
                    self.assertEqual(res["hits"], [])
                with self.subTest("excludeTerms can get rid of part of the term"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string", excludeTerms=["color:red"]
                    )})
                    expected_lexical_facets = {'color': {'red': {'count': 1}}}
                    expected_other_facets = {'color': {'red': {'count': 1}}}
                    res = test_facet_query(
                        retrieval_method=retrieval_method, ranking_method=ranking_method,
                        facets=facets,
                        expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets,
                        filter_string="price:[49 TO 49.1] AND NOT color:red"
                    )
                with self.subTest("excludeTerms work per field"):
                    facets = FacetsParameters(fields={
                        "color": FieldFacetsConfiguration(type="string", excludeTerms=["color:green"]),
                        "brand": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"]),
                    })
                    expected_lexical_facets = {
                        'color': {'red': {'count': 1}},
                        'brand': {'SnugNest': {'count': 1}}
                    }
                    expected_other_facets = {
                        'color': {'red': {'count': 2}},
                        'brand': {'PulseWear': {'count': 1}, 'SnugNest': {'count': 1}, 'SprintX': {'count': 1}}
                    }
                    res = test_facet_query(
                        retrieval_method=retrieval_method, ranking_method=ranking_method,
                        facets=facets,
                        expected_lexical_facets=expected_lexical_facets, expected_other_facets=expected_other_facets,
                        filter_string="color:red AND NOT color:green"
                    )

    def test_facets_filter_string_parsing_edge_cases(self):
        self.add_fashion_docs(EXAMPLE_FASHION_DOCUMENTS)
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                with self.subTest("Filter string with parentheses around the whole string"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string", excludeTerms=["color:red"]
                    )})
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        ),
                        filter="(price:[49 TO 49.1] AND NOT color:red)"
                    )
                    self.assertEqual(res["facets"]["color"]["red"]["count"], 1)
                with self.subTest("Filter string with parentheses around term"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string", excludeTerms=["color:red"]
                    )})
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        ),
                        filter="price:[49 TO 49.1] AND (NOT color:red)"
                    )
                    self.assertEqual(res["facets"]["color"]["red"]["count"], 1)
                with self.subTest("Filter string with parentheses around value"):
                    facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(
                        type="string", excludeTerms=["color:(red)"]
                    )})
                    res = tensor_search.search(
                        config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                        facets=facets,
                        search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                            retrievalMethod=retrieval_method, rankingMethod=ranking_method
                        ),
                        filter="price:[49 TO 49.1] AND NOT color:(red)"
                    )
                    self.assertEqual(res["facets"]["color"]["red"]["count"], 1)

    def test_get_total_hits(self):
        """Test getting total hits for different retrieval methods"""
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                    search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method, rankingMethod=ranking_method
                    ),
                    track_total_hits=True
                )
                if retrieval_method == RetrievalMethod.Lexical:
                    self.assertEqual(res["totalHits"], 1)
                else:
                    self.assertEqual(res["totalHits"], 4)

    def test_get_total_hits_with_facets(self):
        """Test getting total hits with facets returns consistent counts"""
        self.add_fashion_docs()
        facets = FacetsParameters(fields={"color": FieldFacetsConfiguration(type="string")})
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Tensor, RankingMethod.Tensor),
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
                (RetrievalMethod.Disjunction, RankingMethod.RRF),
        ):
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                res = tensor_search.search(
                    config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
                    search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method, rankingMethod=ranking_method
                    ),
                    track_total_hits=True,
                    facets=facets
                )
                # Verify totalHits matches sum of facet counts
                facet_total = sum(v["count"] for v in res["facets"]["color"].values())
                self.assertEqual(res["totalHits"], facet_total)

    def test_get_total_hits_no_matches(self):
        """Test getting total hits when there are no matching documents"""
        self.add_fashion_docs()
        for retrieval_method, ranking_method in (
                (RetrievalMethod.Lexical, RankingMethod.Lexical),
        ):
            res = tensor_search.search(
                config=self.config, index_name=self.semi_structured_default_text_index.name, text="nonexistent",
                search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                    retrievalMethod=retrieval_method, rankingMethod=ranking_method
                ),
                track_total_hits=True
            )
            self.assertEqual(res["totalHits"], 0)
            self.assertEqual(len(res["hits"]), 0)

    def test_get_non_existent_array_field_returns_empty_facet_value(self):
        """
        Test that a non-existent array field in facets raises an error
        """
        self.add_fashion_docs()
        facets = FacetsParameters(fields={"non_existent_field": FieldFacetsConfiguration(type="array")})
        res = tensor_search.search(
            config=self.config, index_name=self.semi_structured_default_text_index.name, text="shirt",
            search_method=SearchMethod.HYBRID, hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Tensor, rankingMethod=RankingMethod.Tensor
            ),
            facets=facets
        )
        self.assertEqual(res["facets"]["non_existent_field"], {})

