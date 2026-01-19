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


if __name__ == '__main__':
    unittest.main()
