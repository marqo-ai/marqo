"""Unit tests for CollapseSearch (src/marqo/core/search/collapse_search.py).

Coverage map:
    CollapseSearch.search():
        - test_search_calls_two_hybrid_searches_and_merges
        - test_search_returns_relevance_results_when_no_collected_ids
        - test_search_raises_internal_error_when_sort_by_missing

    CollapseSearch.collect_parent_ids():
        - test_collect_parent_ids_numeric_filtering (subtests for numeric, non-numeric, missing, mixed, empty, bool, none, negative)
        - test_collect_parent_ids_always_fetch_variants_includes_all

    CollapseSearch.generate_collapse_sort_by_query():
        - test_generate_collapse_sort_by_query_sets_correct_params
        - test_generate_collapse_sort_by_query_filter_string (subtests for multi-id, single-id, escaped ids)
        - test_generate_collapse_sort_by_query_deep_copies_collapse
        - test_generate_collapse_sort_by_query_preserves_and_nullifies_params

    CollapseSearch.merge_two_collapse_results():
        - test_merge_hit_field_handling (subtests for replaced fields, meta preservation, dropped fields, unique sorted fields, highlights, originalId)
        - test_merge_keeps_original_hit_when_not_in_sorted
        - test_merge_edge_cases (subtests for empty sorted, none collapse field, non-hit keys preserved)

    CollapseSortBy:
        - test_sort_by_order_generates_correct_weight (subtests for asc, desc, default)
"""
import unittest
from unittest.mock import MagicMock, patch

from marqo.core.exceptions import InternalError
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.search.collapse_search import CollapseSearch, HybridSearchInternalParameters
from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
from tests.unit_tests.marqo_test import MarqoTestCase


def _make_collapse_search(
        collapse_field="category",
        sort_field="price",
        sort_order="asc",
        always_fetch_variants=False,
        original_attributes_to_retrieve=None,
) -> CollapseSearch:
    """Helper to create a CollapseSearch with mocked internal_params, bypassing __init__ validation."""
    cs = CollapseSearch.__new__(CollapseSearch)

    collapse = CollapseModel(
        name=collapse_field,
        sort_by=CollapseSortBy(
            fields=[CollapseSortByField(fieldName=sort_field, order=sort_order)],
            alwaysFetchVariants=always_fetch_variants,
        )
    )

    mock_params = MagicMock(spec=HybridSearchInternalParameters)
    mock_params.collapse = collapse
    mock_params.config = MagicMock()
    mock_params.marqo_index = MagicMock()
    mock_params.query = "test query"
    mock_params.result_count = 5
    mock_params.offset = 0
    mock_params.rerank_depth = None
    mock_params.ef_search = None
    mock_params.approximate = True
    mock_params.approximate_threshold = None
    mock_params.searchable_attributes = None
    mock_params.filter_string = None
    mock_params.device = None
    mock_params.attributes_to_retrieve = None
    mock_params.boost = None
    mock_params.media_download_headers = None
    mock_params.context = None
    mock_params.score_modifiers = None
    mock_params.model_auth = None
    mock_params.highlights = False
    mock_params.text_query_prefix = None
    mock_params.hybrid_parameters = HybridParameters(
        retrievalMethod=RetrievalMethod.Lexical,
        rankingMethod=RankingMethod.Lexical,
    )
    mock_params.facets = None
    mock_params.track_total_hits = None
    mock_params.language = None
    mock_params.relevance_cutoff = None
    mock_params.sort_by = None
    mock_params.interpolation_method = None
    mock_params.recency_parameters = None

    cs.internal_params = mock_params
    # These are set in __init__ for attributes_to_retrieve handling
    cs.original_attributes_to_retrieve = original_attributes_to_retrieve
    cs.modified_attributes_to_retrieve = None
    return cs


def _patch_internal_params_validation():
    """Patch HybridSearchInternalParameters to skip marqo_index validation."""
    return patch(
        "marqo.core.search.collapse_search.HybridSearchInternalParameters",
        side_effect=lambda **kwargs: MagicMock(**kwargs),
    )


class TestCollapseSearchSearch(MarqoTestCase):
    """Tests for CollapseSearch.search() orchestration."""

    @patch("marqo.core.search.hybrid_search.HybridSearch")
    def test_search_calls_two_hybrid_searches_and_merges(self, MockHybridSearch):
        """Two HybridSearch.execute_search calls: relevance collapse then sorted collapse."""
        cs = _make_collapse_search()

        relevance_results = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
        ]}
        sorted_results = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10},
            {"_id": "h4", "category": "g2", "price": 20},
        ]}

        mock_instance = MockHybridSearch.return_value
        mock_instance.execute_search.side_effect = [relevance_results, sorted_results]

        with patch.object(cs, 'generate_collapse_sort_by_query', return_value=cs.internal_params):
            result = cs.search()

        self.assertEqual(2, mock_instance.execute_search.call_count)
        self.assertEqual(2, len(result["hits"]))
        self.assertEqual(["h3", "h4"], [h["_id"] for h in result["hits"]])

    @patch("marqo.core.search.hybrid_search.HybridSearch")
    def test_search_returns_relevance_results_when_no_collected_ids(self, MockHybridSearch):
        """When no document IDs are collected, returns relevance results directly."""
        cs = _make_collapse_search()

        relevance_results = {"hits": [
            {"_id": "h1", "category": "g1", "price": "expensive", "_score": 0.9},
        ]}
        mock_instance = MockHybridSearch.return_value
        mock_instance.execute_search.return_value = relevance_results

        result = cs.search()

        self.assertEqual(1, mock_instance.execute_search.call_count)
        self.assertIs(result, relevance_results)

    def test_search_raises_internal_error_when_sort_by_missing(self):
        """Raises InternalError if collapse.sort_by is None."""
        cs = CollapseSearch.__new__(CollapseSearch)
        mock_params = MagicMock()
        mock_params.collapse = CollapseModel(name="category")
        cs.internal_params = mock_params

        with self.assertRaises(InternalError):
            cs.search()


class TestCollectParentIds(unittest.TestCase):
    """Tests for CollapseSearch.collect_parent_ids()."""

    def test_collect_parent_ids_numeric_filtering(self):
        """Verifies numeric filtering: includes int/float, excludes string/missing/bool/None."""
        cs = _make_collapse_search()

        test_cases = [
            ("int and float values are collected", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": 100},
                    {"_id": "h2", "category": "g2", "price": 50.5},
                ]
            }, ["g1", "g2"]),
            ("string values are excluded", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": "expensive"},
                    {"_id": "h2", "category": "g2", "price": "cheap"},
                ]
            }, []),
            ("missing sort field is excluded", {
                "hits": [{"_id": "h1", "category": "g1"}]
            }, []),
            ("mixed: only numeric collected", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": 100},
                    {"_id": "h2", "category": "g2", "price": "expensive"},
                    {"_id": "h3", "category": "g3"},
                    {"_id": "h4", "category": "g4", "price": 0},
                ]
            }, ["g1", "g4"]),
            ("empty hits", {"hits": []}, []),
            ("no hits key", {}, []),
            ("boolean values are excluded", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": True},
                    {"_id": "h2", "category": "g2", "price": False},
                ]
            }, []),
            ("booleans excluded, numerics collected in same set", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": True},
                    {"_id": "h2", "category": "g2", "price": 42},
                    {"_id": "h3", "category": "g3", "price": False},
                    {"_id": "h4", "category": "g4", "price": 3.14},
                ]
            }, ["g2", "g4"]),
            ("None value is excluded", {
                "hits": [{"_id": "h1", "category": "g1", "price": None}]
            }, []),
            ("negative numbers are collected", {
                "hits": [
                    {"_id": "h1", "category": "g1", "price": -50},
                    {"_id": "h2", "category": "g2", "price": -0.5},
                ]
            }, ["g1", "g2"]),
        ]

        for description, results, expected in test_cases:
            with self.subTest(description):
                self.assertEqual(expected, cs.collect_parent_ids(results))

    def test_collect_parent_ids_always_fetch_variants_includes_all(self):
        """With always_fetch_variants=True, all hits are collected regardless of field type."""
        cs = _make_collapse_search(always_fetch_variants=True)
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100},
            {"_id": "h2", "category": "g2", "price": "expensive"},
            {"_id": "h3", "category": "g3"},
        ]}
        self.assertEqual(["g1", "g2", "g3"], cs.collect_parent_ids(results))


class TestGenerateCollapseSortByQuery(unittest.TestCase):
    """Tests for CollapseSearch.generate_collapse_sort_by_query()."""

    def test_generate_collapse_sort_by_query_sets_correct_params(self):
        """Verifies query="*", result_count=len(parent_ids), lexical retrieval, execute_sort enabled."""
        cs = _make_collapse_search()
        with _patch_internal_params_validation() as MockParams:
            result_mock = MockParams.side_effect(
                config=cs.internal_params.config, marqo_index=cs.internal_params.marqo_index,
                query="*", result_count=3, offset=0, rerank_depth=None, ef_search=None,
                approximate=True, approximate_threshold=None, searchable_attributes=None,
                filter_string=None, device=None, attributes_to_retrieve=None, boost=None,
                media_download_headers=None, context=None, score_modifiers=None, model_auth=None,
                highlights=False, text_query_prefix=None,
                hybrid_parameters=HybridParameters(retrievalMethod=RetrievalMethod.Lexical,
                                                   rankingMethod=RankingMethod.Lexical),
                facets=None, track_total_hits=False, language=None, relevance_cutoff=None,
                sort_by=None, interpolation_method=None,
                collapse=cs.internal_params.collapse.copy(deep=True), recency_parameters=None,
            )
            result = cs.generate_collapse_sort_by_query(["g1", "g2", "g3"])

        call_kwargs = MockParams.call_args[1]
        self.assertEqual("*", call_kwargs["query"])
        self.assertEqual(3, call_kwargs["result_count"])
        self.assertEqual(0, call_kwargs["offset"])
        self.assertIsNone(call_kwargs["rerank_depth"])
        self.assertEqual(RetrievalMethod.Lexical, call_kwargs["hybrid_parameters"].retrievalMethod)
        self.assertEqual(RankingMethod.Lexical, call_kwargs["hybrid_parameters"].rankingMethod)
        self.assertFalse(call_kwargs["track_total_hits"])

    def test_generate_collapse_sort_by_query_filter_string(self):
        """Verifies filter string format and escaping for various parent ID inputs."""
        cs = _make_collapse_search()

        test_cases = [
            ("multiple parent ids",
             ["parent_a", "parent_b"],
             'category in ("parent_a", "parent_b")'),
            ("single parent id",
             ["only_one"],
             'category in ("only_one")'),
            ("quotes and backslashes escaped",
             ['id_with_"quote', 'id_with_\\backslash'],
             'category in ("id_with_\\"quote", "id_with_\\\\backslash")'),
            ("normal id",
             ['normal_id'],
             'category in ("normal_id")'),
            ("mixed special characters",
             ['a"b\\c"d'],
             'category in ("a\\"b\\\\c\\"d")'),
        ]

        for description, parent_ids, expected_filter in test_cases:
            with self.subTest(description):
                with _patch_internal_params_validation():
                    result = cs.generate_collapse_sort_by_query(parent_ids)
                self.assertTrue(result.collapse.sort_by.should_execute_sort())
                filter_str = result.collapse.sort_by.get_collapse_sort_by_filter_string()
                self.assertEqual(expected_filter, filter_str)

    def test_generate_collapse_sort_by_query_deep_copies_collapse(self):
        """The returned collapse is a deep copy — modifying it doesn't affect the original."""
        cs = _make_collapse_search()
        with _patch_internal_params_validation():
            result = cs.generate_collapse_sort_by_query(["g1"])

        self.assertTrue(result.collapse.sort_by.should_execute_sort())
        self.assertFalse(cs.internal_params.collapse.sort_by.should_execute_sort())

    def test_generate_collapse_sort_by_query_preserves_and_nullifies_params(self):
        """searchable_attributes are preserved; boost, media_download_headers, context, score_modifiers, model_auth are nullified."""
        cs = _make_collapse_search()
        cs.internal_params.searchable_attributes = ["title", "description"]
        cs.internal_params.boost = {"field": 2.0}
        cs.internal_params.media_download_headers = {"Authorization": "Bearer x"}
        cs.internal_params.context = MagicMock()
        cs.internal_params.score_modifiers = MagicMock()
        cs.internal_params.model_auth = MagicMock()

        with _patch_internal_params_validation() as MockParams:
            cs.generate_collapse_sort_by_query(["g1"])

        call_kwargs = MockParams.call_args[1]

        with self.subTest("searchable_attributes preserved"):
            self.assertEqual(["title", "description"], call_kwargs["searchable_attributes"])

        for param in ["boost", "media_download_headers", "context", "score_modifiers", "model_auth"]:
            with self.subTest(f"{param} nullified"):
                self.assertIsNone(call_kwargs[param])


class TestMergeTwoCollapseResults(unittest.TestCase):
    """Tests for CollapseSearch.merge_two_collapse_results()."""

    def test_merge_hit_field_handling(self):
        """Verifies full merge behavior: sorted fields replace, meta preserved from relevance,
        relevance-only non-meta fields dropped, sorted-only fields included, _highlights reset, _originalId added."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {
                "_id": "h1", "category": "g1", "price": 100, "color": "red",
                "_score": 0.9, "_rank": 1, "_pixel_data": "abc",
                "_highlights": [{"title": "match"}], "_recency_score": 0.5,
                "_lexical_score": 0.7, "_tensor_score": 0.3,
            },
        ]}
        sorted_res = {"hits": [
            {
                "_id": "h3", "category": "g1", "price": 10,
                "_score": 0.1, "_rank": 5, "_pixel_data": "def",
                "brand": "Acme", "sku": "X123",
            },
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        expected = {
            # _id from sorted variant
            "_id": "h3",
            # shared non-meta fields from sorted
            "category": "g1",
            "price": 10,
            # fields only in sorted are included
            "brand": "Acme",
            "sku": "X123",
            # meta fields preserved from relevance (not _id, not _highlights)
            "_score": 0.9,
            "_rank": 1,
            "_pixel_data": "abc",
            "_recency_score": 0.5,
            "_lexical_score": 0.7,
            "_tensor_score": 0.3,
            # _highlights reset, _originalId added
            "_highlights": [{}],
            "_originalId": "h1",
            # Note: "color" from relevance is dropped (not in sorted)
        }

        self.assertEqual(expected, result["hits"][0])

    def test_merge_keeps_relevance_hit_when_sort_value_is_tied(self):
        """When sorted variant has the same sort value as relevance hit (tie), keep the relevance hit."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "same1", "category": "g1", "price": 10, "_score": 0.9},
        ]}
        sorted_res = {"hits": [
            {"_id": "same1", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        expected = {
            "_id": "same1",
            "category": "g1",
            "price": 10,
            "_score": 0.9,
        }
        self.assertEqual(expected, result["hits"][0])

    def test_merge_keeps_original_hit_when_not_in_sorted(self):
        """Hits not in sorted results are kept unchanged, without _originalId."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        expected_merged = {
            "_id": "h3",
            "category": "g1",
            "price": 10,
            "_score": 0.9,
            "_highlights": [{}],
            "_originalId": "h1",
        }
        expected_unmatched = {
            "_id": "h2",
            "category": "g2",
            "price": 200,
            "_score": 0.8,
        }
        self.assertEqual(expected_merged, result["hits"][0])
        self.assertEqual(expected_unmatched, result["hits"][1])

    def test_merge_edge_cases(self):
        """Edge cases: empty sorted results, None collapse field, non-hit keys preserved."""
        cs = _make_collapse_search()

        with self.subTest("empty sorted results keeps all relevance hits"):
            relevance = {"hits": [
                {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
                {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
            ]}
            sorted_res = {"hits": []}

            result = cs.merge_two_collapse_results(relevance, sorted_res)
            expected = {"hits": [
                {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
                {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
            ]}
            self.assertEqual(expected, result)

        with self.subTest("sorted hit with None collapse field is skipped"):
            relevance = {"hits": [
                {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            ]}
            sorted_res = {"hits": [
                {"_id": "h3", "category": None, "price": 10},
            ]}

            result = cs.merge_two_collapse_results(relevance, sorted_res)
            expected = {"hits": [
                {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            ]}
            self.assertEqual(expected, result)

        with self.subTest("non-hit keys (totalHits, processingTimeMs) preserved"):
            relevance = {
                "hits": [{"_id": "h1", "category": "g1", "price": 100, "_score": 0.9}],
                "totalHits": 42,
                "processingTimeMs": 15,
            }
            sorted_res = {"hits": [{"_id": "h3", "category": "g1", "price": 10}]}

            result = cs.merge_two_collapse_results(relevance, sorted_res)
            expected = {
                "hits": [{
                    "_id": "h3",
                    "category": "g1",
                    "price": 10,
                    "_score": 0.9,
                    "_highlights": [{}],
                    "_originalId": "h1",
                }],
                "totalHits": 42,
                "processingTimeMs": 15,
            }
            self.assertEqual(expected, result)


class TestMergeAttributesToRetrieve(unittest.TestCase):
    """Tests for merge_two_collapse_results handling of original_attributes_to_retrieve."""

    def test_merge_removes_collapse_and_sort_fields_not_in_original_attributes(self):
        """When original_attributes_to_retrieve excludes collapse/sort_by fields, they are removed from merged hits."""
        cs = _make_collapse_search(
            collapse_field="category",
            sort_field="price",
            original_attributes_to_retrieve=["title"],  # excludes 'category' and 'price'
        )

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "title": "Shoe A", "_score": 0.9},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10, "title": "Shoe B"},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        # 'category' and 'price' should be removed since not in original_attributes_to_retrieve
        expected = {
            "_id": "h3",
            "title": "Shoe B",
            "_score": 0.9,
            "_highlights": [{}],
            "_originalId": "h1",
        }
        self.assertEqual(expected, result["hits"][0])

    def test_merge_keeps_fields_when_in_original_attributes(self):
        """When original_attributes_to_retrieve includes collapse/sort_by fields, they are kept."""
        cs = _make_collapse_search(
            collapse_field="category",
            sort_field="price",
            original_attributes_to_retrieve=["title", "price", "category"],
        )

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "title": "Shoe A", "_score": 0.9},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10, "title": "Shoe B"},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        # 'category' and 'price' should be kept since in original_attributes_to_retrieve
        expected = {
            "_id": "h3",
            "category": "g1",
            "price": 10,
            "title": "Shoe B",
            "_score": 0.9,
            "_highlights": [{}],
            "_originalId": "h1",
        }
        self.assertEqual(expected, result["hits"][0])

    def test_merge_does_not_remove_fields_when_original_attributes_is_none(self):
        """When original_attributes_to_retrieve is None (all fields), nothing is removed."""
        cs = _make_collapse_search(
            collapse_field="category",
            sort_field="price",
            original_attributes_to_retrieve=None,
        )

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "title": "Shoe A", "_score": 0.9},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10, "title": "Shoe B"},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res)

        # All fields should be kept since original_attributes_to_retrieve is None
        expected = {
            "_id": "h3",
            "category": "g1",
            "price": 10,
            "title": "Shoe B",
            "_score": 0.9,
            "_highlights": [{}],
            "_originalId": "h1",
        }
        self.assertEqual(expected, result["hits"][0])


class TestCollapseSortByOrder(unittest.TestCase):
    """Tests for CollapseSortBy.generate_vespa_sort_by_query_input() with different sort orders."""

    def test_sort_by_order_generates_correct_weight(self):
        """Verifies weight generation: desc=1, asc=-1, default=desc."""
        test_cases = [
            ("desc order produces weight 1", "desc", {"price": 1}),
            ("asc order produces weight -1", "asc", {"price": -1}),
            ("default order is desc (weight 1)", None, {"price": 1}),
        ]

        for description, order, expected in test_cases:
            with self.subTest(description):
                kwargs = {"fieldName": "price"}
                if order is not None:
                    kwargs["order"] = order
                sort_by = CollapseSortBy(fields=[CollapseSortByField(**kwargs)])
                self.assertEqual(expected, sort_by.generate_vespa_sort_by_query_input())


if __name__ == "__main__":
    unittest.main()