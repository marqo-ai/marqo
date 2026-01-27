import unittest
from unittest.mock import MagicMock, patch

from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration, RangeConfiguration
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.version import get_version



class TestFacetsTerm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock objects
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def test_string_facet(self):
        """Test basic string facet configuration"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string")
            }
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(100) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_array_facet(self):
        """Test array facet configuration"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="array")
            }
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__string_array_field1) max(100) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_number_facet_without_ranges(self):
        """Test number facet without ranges"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="number")
            }
        )
        result = self.index._get_facets_term(facets)
        expected = ('all(all(group(0) max(100) order(-count()) '
                   'each(output(sum(marqo__int_fields{"field1"}), avg(marqo__int_fields{"field1"}), '
                   'min(marqo__int_fields{"field1"}), max(marqo__int_fields{"field1"}), count()))) '
                   'all(group(-0) max(100) order(-count()) '
                   'each(output(sum(marqo__float_fields{"field1"}), avg(marqo__float_fields{"field1"}), '
                   'min(marqo__float_fields{"field1"}), max(marqo__float_fields{"field1"}), count()))) )')
        self.assertEqual(result, expected)

    def test_number_facet_with_ranges(self):
        """Test number facet with ranges"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(
                    type="number",
                    ranges=[
                        {"from": 0, "to": 10},
                        {"from": 10, "to": 20}
                    ]
                )
            }
        )
        result = self.index._get_facets_term(facets)
        expected = ('all(all(group(predefined(marqo__int_fields{"field1"}, bucket(0.0, 10.0), bucket(10.0, 20.0))) '
                   'max(100) order(-count()) each(output(sum(marqo__int_fields{"field1"}), avg(marqo__int_fields{"field1"}), '
                   'min(marqo__int_fields{"field1"}), max(marqo__int_fields{"field1"}), count()))) '
                   'all(group(predefined(marqo__float_fields{"field1"}, bucket(0.0, 10.0), bucket(10.0, 20.0))) '
                   'max(100) order(-count()) each(output(sum(marqo__float_fields{"field1"}), avg(marqo__float_fields{"field1"}), '
                   'min(marqo__float_fields{"field1"}), max(marqo__float_fields{"field1"}), count()))) )')
        self.assertEqual(result, expected)

    def test_number_facet_with_infinite_ranges(self):
        """Test number facet with infinite ranges"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(
                    type="number",
                    ranges=[
                        {"to": 0},
                        {"from": 0}
                    ]
                )
            }
        )
        result = self.index._get_facets_term(facets)
        expected = ('all(all(group(predefined(marqo__int_fields{"field1"}, bucket(-inf, 0.0), bucket(0.0, inf))) '
                   'max(100) order(-count()) each(output(sum(marqo__int_fields{"field1"}), avg(marqo__int_fields{"field1"}), '
                   'min(marqo__int_fields{"field1"}), max(marqo__int_fields{"field1"}), count()))) '
                   'all(group(predefined(marqo__float_fields{"field1"}, bucket(-inf, 0.0), bucket(0.0, inf))) '
                   'max(100) order(-count()) each(output(sum(marqo__float_fields{"field1"}), avg(marqo__float_fields{"field1"}), '
                   'min(marqo__float_fields{"field1"}), max(marqo__float_fields{"field1"}), count()))) )')
        self.assertEqual(result, expected)

    def test_maxResults_field_level(self):
        """Test maxResults at field level"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(
                    type="string",
                    maxResults=5
                )
            }
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(5) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_maxResults_global_level(self):
        """Test maxResults at global level"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string")
            },
            maxResults=5
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(5) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_order_field_level(self):
        """Test order at field level"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(
                    type="string",
                    order="asc"
                )
            }
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(100) order(count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_order_global_level(self):
        """Test order at global level"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string")
            },
            order="asc"
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(100) order(count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_maxDepth(self):
        """Test maxDepth parameter"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string")
            },
            maxDepth=3
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(max(3) all(group(marqo__short_string_fields{"field1"}) max(100) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_multiple_fields_mixed_types(self):
        """Test multiple fields of different types"""
        facets = FacetsParameters(
            fields={
                "string_field": FieldFacetsConfiguration(type="string"),
                "array_field": FieldFacetsConfiguration(type="array"),
                "number_field": FieldFacetsConfiguration(
                    type="number",
                    ranges=[
                        {"from": 0, "to":10}
                    ]
                )
            },
            maxDepth=2,
            maxResults=5,
            order="desc"
        )
        result = self.index._get_facets_term(facets)
        expected = ('all(max(2) '
                   'all(group(marqo__short_string_fields{"string_field"}) max(5) order(-count()) each(output(count()))) '
                   'all(group(marqo__string_array_array_field) max(5) order(-count()) each(output(count()))) '
                   'all(group(predefined(marqo__int_fields{"number_field"}, bucket(0.0, 10.0))) max(5) order(-count()) '
                   'each(output(sum(marqo__int_fields{"number_field"}), avg(marqo__int_fields{"number_field"}), '
                   'min(marqo__int_fields{"number_field"}), max(marqo__int_fields{"number_field"}), count()))) '
                   'all(group(predefined(marqo__float_fields{"number_field"}, bucket(0.0, 10.0))) max(5) order(-count()) '
                   'each(output(sum(marqo__float_fields{"number_field"}), avg(marqo__float_fields{"number_field"}), '
                   'min(marqo__float_fields{"number_field"}), max(marqo__float_fields{"number_field"}), count()))) )')
        self.assertEqual(result, expected)

    def test_field_and_global_parameters_precedence(self):
        """Test that field-level parameters take precedence over global parameters"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(
                    type="string",
                    maxResults=10,
                    order="asc"
                )
            },
            maxResults=5,
            order="desc"
        )
        result = self.index._get_facets_term(facets)
        expected = 'all(all(group(marqo__short_string_fields{"field1"}) max(10) order(count()) each(output(count()))) )'
        self.assertEqual(result, expected)


class TestFacetsTermExcludeTerms(unittest.TestCase):
    """Tests for exclude_terms functionality in _get_facets_term"""

    @classmethod
    def setUpClass(cls):
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def test_field_with_exclude_terms_none_included_in_main_query(self):
        """Test that a field with exclude_terms=None is included in the main facets query"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=None)
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=None)
        self.assertIsNotNone(result)
        self.assertIn('marqo__short_string_fields{"field1"}', result)

    def test_field_with_exclude_terms_empty_list_included_in_main_query(self):
        """Test that a field with exclude_terms=[] (empty list) is included in the main facets query.

        This is a regression test for the bug where empty list was treated differently from None.
        """
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=[])
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=None)
        self.assertIsNotNone(result)
        self.assertIn('marqo__short_string_fields{"field1"}', result)

    def test_field_with_exclude_terms_skipped_in_main_query(self):
        """Test that a field with exclude_terms is NOT included in the main facets query"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=None)
        # Should return None since no fields are included
        self.assertIsNone(result)

    def test_field_with_exclude_terms_included_in_exclusion_query(self):
        """Test that a field with exclude_terms IS included in the matching exclusion query"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=["color:red"])
        self.assertIsNotNone(result)
        self.assertIn('marqo__short_string_fields{"field1"}', result)

    def test_field_with_exclude_terms_skipped_in_non_matching_exclusion_query(self):
        """Test that a field with exclude_terms is NOT included in a non-matching exclusion query"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=["size:large"])
        # Should return None since field1's exclude_terms don't match
        self.assertIsNone(result)

    def test_mixed_fields_with_and_without_exclude_terms(self):
        """Test mixed fields: some with exclude_terms, some without"""
        facets = FacetsParameters(
            fields={
                "field_no_exclude": FieldFacetsConfiguration(type="string", excludeTerms=None),
                "field_empty_exclude": FieldFacetsConfiguration(type="string", excludeTerms=[]),
                "field_with_exclude": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )

        # Main query should include field_no_exclude and field_empty_exclude, but not field_with_exclude
        result = self.index._get_facets_term(facets, exclusion_terms=None)
        self.assertIsNotNone(result)
        self.assertIn('marqo__short_string_fields{"field_no_exclude"}', result)
        self.assertIn('marqo__short_string_fields{"field_empty_exclude"}', result)
        self.assertNotIn('marqo__short_string_fields{"field_with_exclude"}', result)

    def test_exclusion_query_only_includes_matching_fields(self):
        """Test that exclusion query only includes fields with matching exclude_terms"""
        facets = FacetsParameters(
            fields={
                "field_no_exclude": FieldFacetsConfiguration(type="string", excludeTerms=None),
                "field_empty_exclude": FieldFacetsConfiguration(type="string", excludeTerms=[]),
                "field_with_exclude": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )

        # Exclusion query for ["color:red"] should only include field_with_exclude
        result = self.index._get_facets_term(facets, exclusion_terms=["color:red"])
        self.assertIsNotNone(result)
        self.assertNotIn('marqo__short_string_fields{"field_no_exclude"}', result)
        self.assertNotIn('marqo__short_string_fields{"field_empty_exclude"}', result)
        self.assertIn('marqo__short_string_fields{"field_with_exclude"}', result)

    def test_all_fields_have_exclude_terms_main_query_returns_none(self):
        """Test that main query returns None when all fields have exclude_terms"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"]),
                "field2": FieldFacetsConfiguration(type="string", excludeTerms=["size:large"])
            }
        )
        result = self.index._get_facets_term(facets, exclusion_terms=None)
        self.assertIsNone(result)

    def test_multiple_exclude_terms_exact_match_required(self):
        """Test that fields with multiple exclude_terms require exact set match"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red", "size:large"])
            }
        )

        # Partial match should not include the field
        result_partial = self.index._get_facets_term(facets, exclusion_terms=["color:red"])
        self.assertIsNone(result_partial)

        # Exact match should include the field
        result_exact = self.index._get_facets_term(facets, exclusion_terms=["color:red", "size:large"])
        self.assertIsNotNone(result_exact)
        self.assertIn('marqo__short_string_fields{"field1"}', result_exact)


class TestGetAllFilterTerms(unittest.TestCase):
    """Tests for _get_all_filter_terms helper method"""

    @classmethod
    def setUpClass(cls):
        from marqo.core.search import search_filter
        cls.search_filter = search_filter
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def _create_mock_query_with_filter(self, filter_obj):
        """Helper to create a mock MarqoQuery with a filter"""
        mock_query = MagicMock()
        mock_query.filter = filter_obj
        return mock_query

    def test_no_filter_returns_empty_set(self):
        """Test that None filter returns empty set"""
        mock_query = self._create_mock_query_with_filter(None)
        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, set())

    def test_single_equality_term(self):
        """Test that a single equality term is collected"""
        term = self.search_filter.EqualityTerm("color", "red", "color:red")
        filter_obj = self.search_filter.SearchFilter(term)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, {"color:red"})

    def test_multiple_terms_with_and(self):
        """Test that multiple terms connected with AND are collected"""
        term1 = self.search_filter.EqualityTerm("color", "red", "color:red")
        term2 = self.search_filter.EqualityTerm("size", "large", "size:large")
        and_node = self.search_filter.And(term1, term2)
        filter_obj = self.search_filter.SearchFilter(and_node)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, {"color:red", "size:large"})

    def test_multiple_terms_with_or(self):
        """Test that multiple terms connected with OR are collected"""
        term1 = self.search_filter.EqualityTerm("color", "red", "color:red")
        term2 = self.search_filter.EqualityTerm("color", "blue", "color:blue")
        or_node = self.search_filter.Or(term1, term2)
        filter_obj = self.search_filter.SearchFilter(or_node)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, {"color:red", "color:blue"})

    def test_nested_operators(self):
        """Test that nested operators are traversed correctly"""
        term1 = self.search_filter.EqualityTerm("color", "red", "color:red")
        term2 = self.search_filter.EqualityTerm("size", "large", "size:large")
        term3 = self.search_filter.EqualityTerm("brand", "nike", "brand:nike")
        and_node = self.search_filter.And(term1, term2)
        or_node = self.search_filter.Or(and_node, term3)
        filter_obj = self.search_filter.SearchFilter(or_node)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, {"color:red", "size:large", "brand:nike"})

    def test_not_modifier(self):
        """Test that NOT modifier and its inner term are both collected"""
        term = self.search_filter.EqualityTerm("color", "red", "color:red")
        not_node = self.search_filter.Not(term)
        filter_obj = self.search_filter.SearchFilter(not_node)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        # Both the NOT modifier and the inner term should be collected
        self.assertIn("color:red", result)
        self.assertIn("NOT (color:red)", result)

    def test_range_term(self):
        """Test that range terms are collected"""
        term = self.search_filter.RangeTerm("price", 10, 100, "price:[10 TO 100]")
        filter_obj = self.search_filter.SearchFilter(term)
        mock_query = self._create_mock_query_with_filter(filter_obj)

        result = self.index._get_all_filter_terms(mock_query)
        self.assertEqual(result, {"price:[10 TO 100]"})


class TestGenerateFacetQueriesExcludeTermsFiltering(unittest.TestCase):
    """Tests for exclude_terms filtering in _generate_facet_queries.

    These tests verify that exclude_terms not present in the filter are filtered out
    before constructing separate facets queries.
    """

    @classmethod
    def setUpClass(cls):
        from marqo.core.search import search_filter
        from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod
        cls.search_filter = search_filter
        cls.HybridParameters = HybridParameters
        cls.RetrievalMethod = RetrievalMethod
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.marqo_index.schema_name = "test_schema"
        cls.marqo_index.tensor_fields = []
        cls.marqo_index.lexical_fields = []
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def _create_mock_hybrid_query(self, filter_obj, facets):
        """Helper to create a mock MarqoHybridQuery"""
        mock_query = MagicMock()
        mock_query.filter = filter_obj
        mock_query.facets = facets
        mock_query.track_total_hits = False
        mock_query.collapse_field_name = None
        mock_query.extra_params = {}
        mock_query.hybrid_parameters = self.HybridParameters(
            retrievalMethod=self.RetrievalMethod.Lexical,
            rankingMethod="lexical",
        )
        return mock_query

    def test_exclude_terms_not_in_filter_skipped(self):
        """Test that exclude_terms not in the filter don't generate separate queries"""
        # Filter only has "color:red"
        filter_term = self.search_filter.EqualityTerm("color", "red", "color:red")
        filter_obj = self.search_filter.SearchFilter(filter_term)

        # Facets has exclude_terms=["size:large"] which is NOT in the filter
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["size:large"])
            }
        )

        mock_query = self._create_mock_hybrid_query(filter_obj, facets)

        # Patch _get_lexical_search_term to return a simple term
        with patch.object(self.index, '_get_lexical_search_term', return_value='True'):
            with patch.object(self.index, '_get_tensor_fields_to_search', return_value=[]):
                result = self.index._generate_facet_queries(mock_query)

        # Result should NOT contain the exclude_terms query since "size:large" is not in the filter
        # It should only have the main facets query (if any field passes through)
        queries = result.split("\n---MARQO-YQL-QUERY-DELIMITER---\n")
        # Since field1 has exclude_terms, it won't be in main query either
        # So we should have at most 1 query (the main one with no fields, which returns None)
        # or 0 queries if the main facets term is None
        self.assertLessEqual(len(queries), 1)
        # Importantly, we should NOT have a query that excludes "size:large"
        for query in queries:
            if query:  # Skip empty strings
                self.assertNotIn('size:large', query)

    def test_exclude_terms_in_filter_generates_query(self):
        """Test that exclude_terms present in the filter DO generate separate queries"""
        # Filter has "color:red"
        filter_term = self.search_filter.EqualityTerm("color", "red", "color:red")
        filter_obj = self.search_filter.SearchFilter(filter_term)

        # Facets has exclude_terms=["color:red"] which IS in the filter
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red"])
            }
        )

        mock_query = self._create_mock_hybrid_query(filter_obj, facets)

        with patch.object(self.index, '_get_lexical_search_term', return_value='True'):
            with patch.object(self.index, '_get_tensor_fields_to_search', return_value=[]):
                result = self.index._generate_facet_queries(mock_query)

        # Should have a query that excludes "color:red"
        queries = result.split("\n---MARQO-YQL-QUERY-DELIMITER---\n")
        # Should have at least one query for the exclude_terms
        has_exclusion_query = any(
            'field1' in q for q in queries if q
        )
        self.assertTrue(has_exclusion_query)

    def test_partial_exclude_terms_filtered(self):
        """Test that only valid exclude_terms are used when some are in filter and some aren't"""
        # Filter has "color:red" only
        filter_term = self.search_filter.EqualityTerm("color", "red", "color:red")
        filter_obj = self.search_filter.SearchFilter(filter_term)

        # Facets has exclude_terms with both valid and invalid terms
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="string", excludeTerms=["color:red", "size:large"])
            }
        )

        mock_query = self._create_mock_hybrid_query(filter_obj, facets)

        with patch.object(self.index, '_get_lexical_search_term', return_value='True'):
            with patch.object(self.index, '_get_tensor_fields_to_search', return_value=[]):
                # We need to patch _get_filter_term to verify it's called with only valid terms
                with patch.object(self.index, '_get_filter_term') as mock_get_filter:
                    mock_get_filter.return_value = 'some_filter'
                    with patch.object(self.index, '_get_facets_term', return_value='some_facets'):
                        result = self.index._generate_facet_queries(mock_query)

                    # Check that _get_filter_term was called with only ["color:red"], not ["color:red", "size:large"]
                    calls = [call for call in mock_get_filter.call_args_list
                             if len(call[0]) > 1 or (call[1] and 'exclude_terms' in call[1])]
                    # Find the call with exclude_terms
                    for call in mock_get_filter.call_args_list:
                        args, kwargs = call
                        if len(args) > 1:
                            exclude_terms_arg = args[1]
                            if exclude_terms_arg is not None:
                                # Should only contain "color:red", not "size:large"
                                self.assertEqual(exclude_terms_arg, ["color:red"])

    def test_exclude_terms_with_different_raw_format_not_matched(self):
        """Test that exclude_terms with different raw format don't match filter terms.

        This is a real-world test case where:
        - Filter contains 'tags:(category:Dresses)' with raw='tags:(category:Dresses)'
        - excludeTerms contains 'category:Dresses'
        - These should NOT match because the raw strings are different

        Expected: 2 queries (one for main facets, one for total hits), no separate exclusion query.
        """
        from marqo.core.search.search_filter import MarqoFilterStringParser

        # Parse the real filter string
        filter_string = "available_markets:pe AND collections:(sale) AND any_variant_inventory_available:(true) AND tags:(category:Dresses)"
        parser = MarqoFilterStringParser()
        filter_obj = parser.parse(filter_string)

        # Create facets matching the user's example
        # RangeConfiguration uses 'from' and 'to' as aliases
        facets = FacetsParameters(
            maxDepth=600,
            maxResults=500,
            fields={
                "named_tags": FieldFacetsConfiguration(type="array", excludeTerms=["category:Dresses"]),
                "color": FieldFacetsConfiguration(type="string", excludeTerms=[]),
                "all_sizes_in_stock_array": FieldFacetsConfiguration(type="array"),
                "price": FieldFacetsConfiguration(
                    type="number",
                    ranges=[
                        {"from": 0, "to": 10},
                        {"from": 10, "to": 25},
                        {"from": 25, "to": 50},
                        {"from": 50, "to": 100},
                        {"from": 100, "to": 150},
                        {"from": 150},  # 150 to inf
                    ],
                    excludeTerms=[]
                )
            }
        )

        mock_query = self._create_mock_hybrid_query(filter_obj, facets)
        mock_query.track_total_hits = True
        # Use default separate_total_hits_query=True

        with patch.object(self.index, '_get_lexical_search_term', return_value='True'):
            with patch.object(self.index, '_get_tensor_fields_to_search', return_value=[]):
                result = self.index._generate_facet_queries(mock_query)

        # Split by delimiter to count queries
        queries = result.split("\n---MARQO-YQL-QUERY-DELIMITER---\n")
        queries = [q for q in queries if q.strip()]  # Remove empty strings

        # Should have exactly 2 queries:
        # 1. Main facets query (all fields including named_tags - its exclude_terms are invalid)
        # 2. Total hits query (because track_total_hits=True and separate_total_hits_query=True)
        # NOT 3 queries (no separate exclusion query for "category:Dresses" since it's not in filter)
        self.assertEqual(len(queries), 2,
                         f"Expected 2 queries (main facets + total hits), got {len(queries)}: {queries}")

        # The main facets query (first query) should include ALL fields including named_tags
        # because named_tags' exclude_terms are invalid (not in filter)
        main_facets_query = queries[0]
        self.assertIn('marqo__string_array_named_tags', main_facets_query,
                      "named_tags should be included in main query since its exclude_terms are not in the filter")
        self.assertIn('marqo__short_string_fields{"color"}', main_facets_query)
        self.assertIn('marqo__string_array_all_sizes_in_stock_array', main_facets_query)
        self.assertIn('marqo__int_fields{"price"}', main_facets_query)
        self.assertIn('marqo__float_fields{"price"}', main_facets_query)

    def test_get_all_filter_terms_preserves_raw_format(self):
        """Verify that _get_all_filter_terms preserves the original raw format of filter terms.

        Filter: 'tags:(category:Dresses)' should have raw='tags:(category:Dresses)', not 'category:Dresses'
        """
        from marqo.core.search.search_filter import MarqoFilterStringParser

        filter_string = "tags:(category:Dresses)"
        parser = MarqoFilterStringParser()
        filter_obj = parser.parse(filter_string)

        mock_query = MagicMock()
        mock_query.filter = filter_obj

        result = self.index._get_all_filter_terms(mock_query)

        # The filter term should be 'tags:(category:Dresses)', not 'category:Dresses'
        self.assertIn("tags:(category:Dresses)", result)
        self.assertNotIn("category:Dresses", result)


if __name__ == '__main__':
    unittest.main()
