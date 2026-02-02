"""Unit tests for CollapseSearch (src/marqo/core/search/collapse_search.py).

Coverage map:
    CollapseSearch.search():
        1. search_calls_two_hybrid_searches_and_merges
            - Verifies the two-phase search flow: relevance collapse → collect IDs → generate sort query → sorted collapse → merge
        2. search_returns_relevance_results_when_no_collected_ids
            - When collect_document_ids returns empty, returns relevance results directly (short-circuit)
        3. search_raises_internal_error_when_sort_by_missing
            - Raises InternalError if collapse.sort_by is None

    CollapseSearch.collect_document_ids():
        4. collect_document_ids_includes_numeric_values
            - Hits with int/float sort field values are collected
        5. collect_document_ids_excludes_non_numeric_values
            - Hits with string sort field values are excluded
        6. collect_document_ids_excludes_missing_field
            - Hits missing the sort field are excluded
        7. collect_document_ids_mixed_numeric_and_non_numeric
            - Only numeric-field hits are collected from a mixed set
        8. collect_document_ids_always_fetch_variants_includes_all
            - With always_fetch_variants=True, all hits are collected regardless of field type
        9. collect_document_ids_empty_hits
            - Returns empty list when search results have no hits

    CollapseSearch.generate_collapse_sort_by_query():
        10. generate_collapse_sort_by_query_sets_correct_params
            - Verifies query="*", result_count=len(parent_ids), lexical retrieval, execute_sort enabled
        11. generate_collapse_sort_by_query_builds_filter_string
            - Verifies the collapse filter string format: 'field in ("id1", "id2")'
        12. generate_collapse_sort_by_query_deep_copies_collapse
            - The returned collapse is a deep copy (modifying it doesn't affect the original)

    CollapseSearch.merge_two_collapse_results():
        13. merge_replaces_hits_from_sorted_results
            - Hits matching sorted results get replaced with sorted variant data
        14. merge_keeps_meta_fields_from_relevance_hit
            - Meta fields (_score, _rank) come from relevance hit, not sorted hit
        15. merge_keeps_original_hit_when_not_in_sorted
            - Hits not in sorted results are kept unchanged
        16. merge_sets_empty_highlights
            - Merged hits always get _highlights=[{}]
        17. merge_handles_empty_sorted_results
            - When sorted results have no hits, all relevance hits are kept as-is
        18. merge_sorted_hit_missing_field_falls_back_to_relevance
            - When sorted hit lacks a field, falls back to relevance hit value via .get(key, value)
        19. merge_skips_sorted_hit_with_none_collapse_field
            - Sorted hits with None collapse field value are not indexed in lookup map
        20. merge_preserves_non_hit_keys_from_relevance_results
            - Non-hit keys (e.g., totalHits, processingTimeMs) are preserved in merged output

    CollapseSearch.collect_document_ids():
        21. collect_document_ids_excludes_boolean_values
            - Boolean values are not int/float instances (bool is subclass of int in Python, so True/False ARE collected)
        22. collect_document_ids_excludes_none_value
            - None sort field value is excluded
        23. collect_document_ids_negative_numbers
            - Negative numbers are collected as valid numeric values

    CollapseSearch.generate_collapse_sort_by_query():
        24. generate_collapse_sort_by_query_preserves_searchable_attributes
            - searchable_attributes from original params are passed through
        25. generate_collapse_sort_by_query_nullifies_optional_params
            - boost, media_download_headers, context, score_modifiers, model_auth are set to None
        26. generate_collapse_sort_by_query_single_parent_id
            - Works with a single parent ID
        27. merge_sets_original_id_from_relevance_hit
            - _originalId is set to the relevance hit's _id on merged hits
        28. merge_original_id_equals_id_when_same_variant_selected
            - When sorted variant has same _id as relevance hit, _originalId still equals relevance _id
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
        """1. Two HybridSearch.execute_search calls: relevance collapse then sorted collapse."""
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

        # Patch generate_collapse_sort_by_query to avoid HybridSearchInternalParameters validation
        with patch.object(cs, 'generate_collapse_sort_by_query', return_value=cs.internal_params):
            result = cs.search()

        self.assertEqual(2, mock_instance.execute_search.call_count)
        self.assertEqual(2, len(result["hits"]))
        # Merged hits should have sorted variant IDs
        self.assertEqual(["h3", "h4"], [h["_id"] for h in result["hits"]])

    @patch("marqo.core.search.hybrid_search.HybridSearch")
    def test_search_returns_relevance_results_when_no_collected_ids(self, MockHybridSearch):
        """2. When no document IDs are collected, returns relevance results directly."""
        cs = _make_collapse_search()

        # All hits have string price values → not collected without always_fetch_variants
        relevance_results = {"hits": [
            {"_id": "h1", "category": "g1", "price": "expensive", "_score": 0.9},
        ]}
        mock_instance = MockHybridSearch.return_value
        mock_instance.execute_search.return_value = relevance_results

        result = cs.search()

        # Only one call (relevance), no second call (sorted)
        self.assertEqual(1, mock_instance.execute_search.call_count)
        self.assertIs(result, relevance_results)

    def test_search_raises_internal_error_when_sort_by_missing(self):
        """3. Raises InternalError if collapse.sort_by is None."""
        cs = CollapseSearch.__new__(CollapseSearch)
        mock_params = MagicMock()
        mock_params.collapse = CollapseModel(name="category")
        cs.internal_params = mock_params

        with self.assertRaises(InternalError):
            cs.search()


class TestCollectDocumentIds(unittest.TestCase):
    """Tests for CollapseSearch.collect_document_ids()."""

    def test_collect_document_ids_includes_numeric_values(self):
        """4. Hits with int/float sort field values are collected."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100},
            {"_id": "h2", "category": "g2", "price": 50.5},
        ]}
        self.assertEqual(["g1", "g2"], cs.collect_parent_ids(results))

    def test_collect_document_ids_excludes_non_numeric_values(self):
        """5. Hits with string sort field values are excluded."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": "expensive"},
            {"_id": "h2", "category": "g2", "price": "cheap"},
        ]}
        self.assertEqual([], cs.collect_parent_ids(results))

    def test_collect_document_ids_excludes_missing_field(self):
        """6. Hits missing the sort field are excluded."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1"},
        ]}
        self.assertEqual([], cs.collect_parent_ids(results))

    def test_collect_document_ids_mixed_numeric_and_non_numeric(self):
        """7. Only numeric-field hits are collected from a mixed set."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100},
            {"_id": "h2", "category": "g2", "price": "expensive"},
            {"_id": "h3", "category": "g3"},
            {"_id": "h4", "category": "g4", "price": 0},
        ]}
        self.assertEqual(["g1", "g4"], cs.collect_parent_ids(results))

    def test_collect_document_ids_always_fetch_variants_includes_all(self):
        """8. With always_fetch_variants=True, all hits are collected regardless of field type."""
        cs = _make_collapse_search(always_fetch_variants=True)
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100},
            {"_id": "h2", "category": "g2", "price": "expensive"},
            {"_id": "h3", "category": "g3"},
        ]}
        self.assertEqual(["g1", "g2", "g3"], cs.collect_parent_ids(results))

    def test_collect_document_ids_empty_hits(self):
        """9. Returns empty list when search results have no hits."""
        cs = _make_collapse_search()
        self.assertEqual([], cs.collect_parent_ids({"hits": []}))
        self.assertEqual([], cs.collect_parent_ids({}))


class TestGenerateCollapseSortByQuery(unittest.TestCase):
    """Tests for CollapseSearch.generate_collapse_sort_by_query()."""

    def test_generate_collapse_sort_by_query_sets_correct_params(self):
        """10. Verifies query="*", result_count=len(parent_ids), lexical retrieval, execute_sort enabled."""
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
            # Verify the mock was called with the right key params
            result = cs.generate_collapse_sort_by_query(["g1", "g2", "g3"])

        call_kwargs = MockParams.call_args[1]
        self.assertEqual("*", call_kwargs["query"])
        self.assertEqual(3, call_kwargs["result_count"])
        self.assertEqual(0, call_kwargs["offset"])
        self.assertIsNone(call_kwargs["rerank_depth"])
        self.assertEqual(RetrievalMethod.Lexical, call_kwargs["hybrid_parameters"].retrievalMethod)
        self.assertEqual(RankingMethod.Lexical, call_kwargs["hybrid_parameters"].rankingMethod)
        self.assertFalse(call_kwargs["track_total_hits"])

    def test_generate_collapse_sort_by_query_builds_filter_string(self):
        """11. Verifies the collapse filter string format."""
        cs = _make_collapse_search()
        with _patch_internal_params_validation():
            result = cs.generate_collapse_sort_by_query(["parent_a", "parent_b"])

        # The mock returns a MagicMock with collapse set from kwargs, but collapse is a deep copy
        # so we verify by checking the collapse object passed to the constructor
        collapse_arg = result.collapse
        self.assertTrue(collapse_arg.sort_by.should_execute_sort())
        filter_str = collapse_arg.sort_by.get_collapse_sort_by_filter_string()
        self.assertEqual('category in ("parent_a", "parent_b")', filter_str)

    def test_generate_collapse_sort_by_query_escapes_parent_ids(self):
        """Parent IDs with special characters (quotes, backslashes) are escaped in the filter string."""
        cs = _make_collapse_search()
        test_cases = [
            (
                ['id_with_"quote', 'id_with_\\backslash'],
                'category in ("id_with_\\"quote", "id_with_\\\\backslash")',
            ),
            (
                ['normal_id'],
                'category in ("normal_id")',
            ),
            (
                ['a"b\\c"d'],
                'category in ("a\\"b\\\\c\\"d")',
            ),
        ]
        for parent_ids, expected_filter in test_cases:
            with self.subTest(parent_ids=parent_ids):
                with _patch_internal_params_validation():
                    result = cs.generate_collapse_sort_by_query(parent_ids)
                filter_str = result.collapse.sort_by.get_collapse_sort_by_filter_string()
                self.assertEqual(expected_filter, filter_str)

    def test_generate_collapse_sort_by_query_deep_copies_collapse(self):
        """12. The returned collapse is a deep copy — modifying it doesn't affect the original."""
        cs = _make_collapse_search()
        with _patch_internal_params_validation():
            result = cs.generate_collapse_sort_by_query(["g1"])

        # The generated query's collapse has execute_sort enabled
        self.assertTrue(result.collapse.sort_by.should_execute_sort())
        # But the original should remain unmodified
        self.assertFalse(cs.internal_params.collapse.sort_by.should_execute_sort())


class TestMergeTwoCollapseResults(unittest.TestCase):
    """Tests for CollapseSearch.merge_two_collapse_results()."""

    def test_merge_replaces_hits_from_sorted_results(self):
        """13. Hits matching sorted results get replaced with sorted variant data."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9, "_highlights": [{"title": "hi"}]},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10, "_score": 0.1},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        merged_hit = result["hits"][0]

        self.assertEqual("h3", merged_hit["_id"])
        self.assertEqual(10, merged_hit["price"])
        self.assertEqual("h1", merged_hit["_originalId"])

    def test_merge_keeps_meta_fields_from_relevance_hit(self):
        """14. Meta fields (_score) come from relevance hit, not sorted hit."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9, "_rank": 1},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10, "_score": 0.1, "_rank": 5},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        merged_hit = result["hits"][0]

        # _score and _rank are meta fields (start with _ and not _id/_highlights) → kept from relevance
        self.assertEqual(0.9, merged_hit["_score"])
        self.assertEqual(1, merged_hit["_rank"])

    def test_merge_keeps_original_hit_when_not_in_sorted(self):
        """15. Hits not in sorted results are kept unchanged."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])

        # g1 was replaced, g2 was not
        self.assertEqual("h3", result["hits"][0]["_id"])
        self.assertEqual("h2", result["hits"][1]["_id"])
        self.assertEqual(200, result["hits"][1]["price"])

    def test_merge_sets_empty_highlights(self):
        """16. Merged hits always get _highlights=[{}]."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9, "_highlights": [{"title": "match"}]},
        ]}
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        self.assertEqual([{}], result["hits"][0]["_highlights"])

    def test_merge_handles_empty_sorted_results(self):
        """17. When sorted results have no hits, all relevance hits are kept as-is."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
            {"_id": "h2", "category": "g2", "price": 200, "_score": 0.8},
        ]}
        sorted_res = {"hits": []}

        result = cs.merge_two_collapse_results(relevance, sorted_res, [])

        self.assertEqual(["h1", "h2"], [h["_id"] for h in result["hits"]])

    def test_merge_sets_original_id_from_relevance_hit(self):
        """27. _originalId is set to the relevance hit's _id on merged hits."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "rel1", "category": "g1", "price": 100, "_score": 0.9},
            {"_id": "rel2", "category": "g2", "price": 200, "_score": 0.8},
        ]}
        sorted_res = {"hits": [
            {"_id": "sorted1", "category": "g1", "price": 10},
            {"_id": "sorted2", "category": "g2", "price": 20},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1", "g2"])

        self.assertEqual("sorted1", result["hits"][0]["_id"])
        self.assertEqual("rel1", result["hits"][0]["_originalId"])
        self.assertEqual("sorted2", result["hits"][1]["_id"])
        self.assertEqual("rel2", result["hits"][1]["_originalId"])

    def test_merge_original_id_equals_id_when_same_variant_selected(self):
        """28. When sorted picks the same doc as relevance, _originalId equals _id."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "same1", "category": "g1", "price": 10, "_score": 0.9},
        ]}
        sorted_res = {"hits": [
            {"_id": "same1", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        merged_hit = result["hits"][0]

        self.assertEqual("same1", merged_hit["_id"])
        self.assertEqual("same1", merged_hit["_originalId"])

    def test_merge_no_original_id_on_unmerged_hits(self):
        """Hits not in sorted results (kept as-is) should NOT have _originalId."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
        ]}
        sorted_res = {"hits": []}

        result = cs.merge_two_collapse_results(relevance, sorted_res, [])
        self.assertNotIn("_originalId", result["hits"][0])

    def test_merge_sorted_hit_missing_field_falls_back_to_relevance(self):
        """18. When sorted hit lacks a field, falls back to relevance hit value via .get(key, value)."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "color": "red", "_score": 0.9},
        ]}
        # Sorted hit has category but no "color" field
        sorted_res = {"hits": [
            {"_id": "h3", "category": "g1", "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        merged_hit = result["hits"][0]

        self.assertEqual("h3", merged_hit["_id"])
        self.assertEqual(10, merged_hit["price"])
        # "color" falls back to relevance value
        self.assertEqual("red", merged_hit["color"])

    def test_merge_skips_sorted_hit_with_none_collapse_field(self):
        """19. Sorted hits with None collapse field value are not indexed in lookup map."""
        cs = _make_collapse_search()

        relevance = {"hits": [
            {"_id": "h1", "category": "g1", "price": 100, "_score": 0.9},
        ]}
        # Sorted hit has None for collapse field
        sorted_res = {"hits": [
            {"_id": "h3", "category": None, "price": 10},
        ]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])
        # g1 not found in sorted lookup → original kept
        self.assertEqual("h1", result["hits"][0]["_id"])

    def test_merge_preserves_non_hit_keys_from_relevance_results(self):
        """20. Non-hit keys (totalHits, processingTimeMs) are preserved in merged output."""
        cs = _make_collapse_search()

        relevance = {
            "hits": [{"_id": "h1", "category": "g1", "price": 100, "_score": 0.9}],
            "totalHits": 42,
            "processingTimeMs": 15,
        }
        sorted_res = {"hits": [{"_id": "h3", "category": "g1", "price": 10}]}

        result = cs.merge_two_collapse_results(relevance, sorted_res, ["g1"])

        self.assertEqual(42, result["totalHits"])
        self.assertEqual(15, result["processingTimeMs"])


class TestCollectDocumentIdsAdditional(unittest.TestCase):
    """Additional tests for CollapseSearch.collect_document_ids()."""

    def test_collect_document_ids_boolean_values_are_excluded(self):
        """21. Boolean values are explicitly excluded even though bool is a subclass of int in Python."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": True},
            {"_id": "h2", "category": "g2", "price": False},
        ]}
        # Booleans are explicitly excluded by the value_is_valid_number check
        self.assertEqual([], cs.collect_parent_ids(results))

    def test_collect_document_ids_boolean_mixed_with_numeric(self):
        """Boolean values are excluded while numeric values in the same result set are collected."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": True},
            {"_id": "h2", "category": "g2", "price": 42},
            {"_id": "h3", "category": "g3", "price": False},
            {"_id": "h4", "category": "g4", "price": 3.14},
        ]}
        self.assertEqual(["g2", "g4"], cs.collect_parent_ids(results))

    def test_collect_document_ids_excludes_none_value(self):
        """22. None sort field value is excluded."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": None},
        ]}
        self.assertEqual([], cs.collect_parent_ids(results))

    def test_collect_document_ids_negative_numbers(self):
        """23. Negative numbers are collected as valid numeric values."""
        cs = _make_collapse_search()
        results = {"hits": [
            {"_id": "h1", "category": "g1", "price": -50},
            {"_id": "h2", "category": "g2", "price": -0.5},
        ]}
        self.assertEqual(["g1", "g2"], cs.collect_parent_ids(results))


class TestCollapseSortByOrder(unittest.TestCase):
    """Tests for CollapseSortBy.generate_vespa_sort_by_query_input() with different sort orders."""

    def test_sort_by_desc_generates_positive_weight(self):
        """Desc order produces weight 1 (higher values preferred)."""
        sort_by = CollapseSortBy(
            fields=[CollapseSortByField(fieldName="price", order="desc")]
        )
        self.assertEqual({"price": 1}, sort_by.generate_vespa_sort_by_query_input())

    def test_sort_by_asc_generates_negative_weight(self):
        """Asc order produces weight -1 (lower values preferred)."""
        sort_by = CollapseSortBy(
            fields=[CollapseSortByField(fieldName="price", order="asc")]
        )
        self.assertEqual({"price": -1}, sort_by.generate_vespa_sort_by_query_input())

    def test_sort_by_default_order_is_desc(self):
        """Default order is desc when not specified."""
        sort_by = CollapseSortBy(
            fields=[CollapseSortByField(fieldName="price")]
        )
        self.assertEqual({"price": 1}, sort_by.generate_vespa_sort_by_query_input())


class TestGenerateCollapseSortByQueryAdditional(unittest.TestCase):
    """Additional tests for CollapseSearch.generate_collapse_sort_by_query()."""

    def test_generate_collapse_sort_by_query_preserves_searchable_attributes(self):
        """24. searchable_attributes from original params are passed through."""
        cs = _make_collapse_search()
        cs.internal_params.searchable_attributes = ["title", "description"]

        with _patch_internal_params_validation() as MockParams:
            cs.generate_collapse_sort_by_query(["g1"])

        call_kwargs = MockParams.call_args[1]
        self.assertEqual(["title", "description"], call_kwargs["searchable_attributes"])

    def test_generate_collapse_sort_by_query_nullifies_optional_params(self):
        """25. boost, media_download_headers, context, score_modifiers, model_auth are set to None."""
        cs = _make_collapse_search()
        # Set non-None values on original params
        cs.internal_params.boost = {"field": 2.0}
        cs.internal_params.media_download_headers = {"Authorization": "Bearer x"}
        cs.internal_params.context = MagicMock()
        cs.internal_params.score_modifiers = MagicMock()
        cs.internal_params.model_auth = MagicMock()

        with _patch_internal_params_validation() as MockParams:
            cs.generate_collapse_sort_by_query(["g1"])

        call_kwargs = MockParams.call_args[1]
        self.assertIsNone(call_kwargs["boost"])
        self.assertIsNone(call_kwargs["media_download_headers"])
        self.assertIsNone(call_kwargs["context"])
        self.assertIsNone(call_kwargs["score_modifiers"])
        self.assertIsNone(call_kwargs["model_auth"])

    def test_generate_collapse_sort_by_query_single_parent_id(self):
        """26. Works with a single parent ID."""
        cs = _make_collapse_search()

        with _patch_internal_params_validation() as MockParams:
            result = cs.generate_collapse_sort_by_query(["only_one"])

        call_kwargs = MockParams.call_args[1]
        self.assertEqual(1, call_kwargs["result_count"])
        filter_str = result.collapse.sort_by.get_collapse_sort_by_filter_string()
        self.assertEqual('category in ("only_one")', filter_str)


if __name__ == "__main__":
    unittest.main()
