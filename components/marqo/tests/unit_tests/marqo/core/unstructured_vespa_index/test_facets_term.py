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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(100) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_array_facet(self):
        """Test array facet configuration"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="array")
            }
        )
        result = self.index._get_facets_term(facets)
        expected = 'all( all(group(marqo__string_array_field1) max(100) order(-count()) each(output(count()))) )'
        self.assertEqual(result, expected)

    def test_number_facet_without_ranges(self):
        """Test number facet without ranges"""
        facets = FacetsParameters(
            fields={
                "field1": FieldFacetsConfiguration(type="number")
            }
        )
        result = self.index._get_facets_term(facets)
        expected = ('all( all(group(0) max(100) order(-count()) '
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
        expected = ('all( all(group(predefined(marqo__int_fields{"field1"}, bucket(0.0, 10.0), bucket(10.0, 20.0))) '
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
        expected = ('all( all(group(predefined(marqo__int_fields{"field1"}, bucket(-inf, 0.0), bucket(0.0, inf))) '
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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(5) order(-count()) each(output(count()))) )'
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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(5) order(-count()) each(output(count()))) )'
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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(100) order(count()) each(output(count()))) )'
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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(100) order(count()) each(output(count()))) )'
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
        expected = 'all( max(3) all(group(marqo__short_string_fields{"field1"}) max(100) order(-count()) each(output(count()))) )'
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
        expected = ('all( max(2) '
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
        expected = 'all( all(group(marqo__short_string_fields{"field1"}) max(10) order(count()) each(output(count()))) )'
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
