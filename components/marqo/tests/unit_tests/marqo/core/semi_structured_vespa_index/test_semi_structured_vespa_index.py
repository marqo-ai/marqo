from unittest.mock import patch

from marqo.core.models import MarqoQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.exceptions import InvalidArgumentError
from tests.unit_tests.marqo_test import MarqoTestCase


class TestSemiStructuredVespaIndex(MarqoTestCase):
    def setUp(self):
        self.vespa_index = SemiStructuredVespaIndex(self.semi_structured_marqo_index(
            name='index1',
            lexical_field_names=['title', 'desc'],
            tensor_field_names=['title'],
            string_array_field_names=['tags', 'markets']
        ))

        self.vespa_index_prior_to_216 = SemiStructuredVespaIndex(self.semi_structured_marqo_index(
            name='index1',
            lexical_field_names=['title', 'desc'],
            tensor_field_names=['title'],
            marqo_version='2.15.0'  # no individual string arrays
        ))

    def test_get_filter_string_escaped_characters(self):
        r"""
        Ensure the \ character is added to vespa query before all special characters (\ and ")
        """
        test_cases = [
            # Equality terms
            # no escaped characters
            ('title:hello',
             'key contains "title", value contains "hello"'),
            # Unescaped backslash gets ignored (double quote does not need to be escaped by user)
            ('title:hel"l\\o',
             'key contains "title", value contains "hel\\"lo"'),
            # Escaped backslash is also escaped in vespa query
            ('title:hel\\"l\\\\o',
             'key contains "title", value contains "hel\\"l\\\\o"'),
             ('ti\\"t\\\\le:hello',
              'key contains "ti\\"t\\\\le", value contains "hello"'),
            # Range terms
              ('nu\\"m\\\\ber:[1 TO 100]',
               'key contains "nu\\"m\\\\ber", value >= 1, value <= 100'),
        ]

        for filter_string, expected_result in test_cases:
            with self.subTest(msg=f"Testing filter string: {filter_string}"):
                marqo_query = MarqoQuery(
                    index_name=self.vespa_index._marqo_index.name,
                    limit=10,
                    filter=filter_string,
                    score_modifiers=[],
                    expose_facets=False
                )
                result_filter_string = self.vespa_index._get_filter_term(marqo_query)
                self.assertIn(expected_result, result_filter_string,)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_vespa_index.SemiStructuredVespaIndex.get_marqo_index')
    def test_get_filter_string_collapse_field(self, mock_get_marqo_index):
        """
        Test that collapse fields use direct attribute filtering instead of standard filtering logic
        """
        # Mock the is_collapse_field method
        mock_index = mock_get_marqo_index.return_value
        mock_index.is_collapse_field.side_effect = lambda field: field in ['parent_id', 'variant_id']
        
        test_cases = [
            # Collapse field filtering - should use direct attribute filter
            ('parent_id:group_1', '(parent_id contains "group_1")'),
            # Test it works on different collapse field name
            ('variant_id:product_123', '(variant_id contains "product_123")'),
            # Escape special characters in collapse field values - quotes need escaping  
            ('parent_id:group"test', '(parent_id contains "group\\"test")'),
            # non-collapse field works differently
            ('color:red', '((marqo__short_string_fields contains sameElement(key contains "color", value contains "red")))'),
        ]
        
        for filter_string, expected_result in test_cases:
            with self.subTest(filter_string=filter_string):
                marqo_query = MarqoQuery(
                    index_name=self.vespa_index._marqo_index.name,
                    limit=10,
                    filter=filter_string,
                    score_modifiers=[],
                    expose_facets=False
                )
                result_filter_string = self.vespa_index._get_filter_term(marqo_query)
                self.assertEqual(expected_result, result_filter_string)

    def test_get_filter_string_equality_paths(self):
        """Test equality filter paths: _id, bool, string array, and float numeric."""
        test_cases = [
            # _id filter
            ('_id:doc123', '(marqo__id contains "doc123")'),
            # Bool filter
            ('title:true',
             '((marqo__bool_fields contains sameElement(key contains "title", value = 1)) OR '
             '(marqo__short_string_fields contains sameElement(key contains "title", value contains "true")))'),
            # String array filter
            ('tags:foo',
             '((marqo__short_string_fields contains sameElement(key contains "tags", value contains "foo")) OR '
             '(marqo__string_array_tags contains "foo"))'),
            # Float numeric filter
            ('title:3.14',
             '((marqo__short_string_fields contains sameElement(key contains "title", value contains "3.14")) OR '
             '(marqo__float_fields contains sameElement(key contains "title", value = 3.14)))'),
        ]

        for filter_string, expected_result in test_cases:
            with self.subTest(filter_string=filter_string):
                marqo_query = MarqoQuery(
                    index_name=self.vespa_index._marqo_index.name,
                    limit=10,
                    filter=filter_string,
                    score_modifiers=[],
                    expose_facets=False
                )
                result_filter_string = self.vespa_index._get_filter_term(marqo_query)
                self.assertEqual(expected_result, result_filter_string)

    def test_get_filter_string_contains(self):
        """Test CONTAINS filter generates correct Vespa syntax for lexical fields."""
        marqo_query = MarqoQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            filter='title CONTAINS hello',
            score_modifiers=[],
            expose_facets=False
        )
        result_filter_string = self.vespa_index._get_filter_term(marqo_query)
        self.assertEqual('(marqo__lexical_title contains "hello")', result_filter_string)

    def test_get_filter_string_contains_nonexistent_field_raises_error(self):
        """Test CONTAINS filter raises error for a field not in the index."""
        marqo_query = MarqoQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            filter='nonexistent CONTAINS hello',
            score_modifiers=[],
            expose_facets=False
        )
        with self.assertRaises(InvalidArgumentError):
            self.vespa_index._get_filter_term(marqo_query)

    def test_vespa_to_marqo_conversion_should_handle_all_fields_from_search_result(self):
        vespa_doc = {
            "id": "index:index1/1/123",
            "relevance": 0.01,
            "fields": {
                "marqo__id": "123",
                # TODO field type is returned from the search result but never used
                "marqo__field_types": {
                    "int_field1": "int"  # ignore the rest since this field is not populated to marqo doc
                },
                "marqo__raw_tensor_score": 0.8,
                "marqo__raw_lexical_score": 0.5,
                "marqo__int_fields": {"int_field1": 1, "int_field2": 2, "int_map1.a": 3, "int_map1.b": 4},
                "marqo__float_fields": {"float_field1": 1.0, "float_map1.a": 2.0, "float_map1.b": 3.0},
                "marqo__bool_fields": {"bool_field1": 1, "bool_field2": 0},
                "marqo__string_array_tags": ["foo", "bar"],
                "marqo__string_array_markets": ["fr", "au"],
                "title": "some product",
                "desc": "some awesome product",
                "matchfeatures": {
                    "closest(marqo__embeddings_title)": {"type": "tensor<float>(p{})", "cells": {"0": 1.0}},
                    "distance(field,marqo__embeddings_title)": 0.3308038115501404,
                    "global_add_modifier": 0.0,
                    "global_mult_modifier": 1.0
                },
                "marqo__chunks_title": ["some product"]
            }
        }

        marqo_doc = self.vespa_index.to_marqo_document(vespa_doc)
        self.assertEqual({
            "_id": "123",
            "_tensor_score": 0.8,
            "_lexical_score": 0.5,
            "int_field1": 1,
            "int_field2": 2,
            "int_map1.a": 3,  # int map is flattened
            "int_map1.b": 4,
            "float_field1": 1.0,
            "float_map1.a": 2.0,  # float map is flattened
            "float_map1.b": 3.0,
            "bool_field1": True,
            "bool_field2": False,
            "tags": ["foo", "bar"],
            "markets": ["fr", "au"],
            "title": "some product",
            "desc": "some awesome product",
            "marqo__tensors": {"title": {"chunks": ["some product"]}}
        }, marqo_doc)

    def test_vespa_to_marqo_conversion_should_handle_highlights(self):
        vespa_doc = {
            "id": "index:index1/1/123",
            "relevance": 0.01,
            "fields": {
                "marqo__id": "123",
                "marqo__raw_tensor_score": 0.8,
                "marqo__raw_lexical_score": 0.5,
                "title": "some product",
                "matchfeatures": {
                    "closest(marqo__embeddings_title)": {"type": "tensor<float>(p{})", "cells": {"0": 1.0}},
                    "distance(field,marqo__embeddings_title)": 0.3308038115501404,
                    "global_add_modifier": 0.0,
                    "global_mult_modifier": 1.0
                },
                "marqo__chunks_title": ["some product"]
            }
        }

        marqo_doc = self.vespa_index.to_marqo_document(vespa_doc, return_highlights=True)
        self.assertEqual({
            "_id": "123",
            "_tensor_score": 0.8,
            "_lexical_score": 0.5,
            "title": "some product",
            "_highlights": [{"title": "some product"}],
            "marqo__tensors": {"title": {"chunks": ["some product"]}}
        }, marqo_doc)

    def test_vespa_to_marqo_conversion_should_convert_all_fields_from_get_document_result(self):
        vespa_doc = {
            "id": "index:index1/1/123",
            "fields": {
                "marqo__id": "123",
                "marqo__version_uuid": "uuid1234",
                "marqo__field_types": {
                    "int_field1": "int"  # ignore the rest since this field is not populated to marqo doc
                },
                "marqo__int_fields": {"int_field1": 1, "int_field2": 2, "int_map1.a": 3, "int_map1.b": 4},
                "marqo__float_fields": {"float_field1": 1.0, "float_map1.a": 2.0, "float_map1.b": 3.0},
                "marqo__bool_fields": {"bool_field1": 1, "bool_field2": 0},
                "marqo__string_array_tags": ["foo", "bar"],
                "marqo__string_array_markets": ["fr", "au"],
                "marqo__lexical_title": "some product",  # get_document has lexical fields returned with the prefix
                "marqo__lexical_desc": "some awesome product",
                "marqo__short_string_fields": {"title": "some product", "desc": "some awesome product"},
                "marqo__chunks_title": ["some product"],
                "marqo__embeddings_title": {"blocks": {"0": [1.0, 2.0]}},
                "marqo__score_modifiers": {'cells': {'int_field1': 1, 'int_map1.a': 3, 'int_map1.b': 4}, 'type': 'tensor(p{})'},
                "marqo__multimodal_params": {'multi_modal': '{"weights": {"title": 1.0, "desc": 0.5}, "type": "multimodal_combination"}'},
                "marqo__chunks_multi_modal": ['{"title": "Test document 2", "desc": "desc1"}'],
                "marqo__embeddings_multi_modal": {"blocks": {"0": [2.0, 3.0]}},
                "marqo__vector_count": 1
            }
        }

        marqo_doc = self.vespa_index.to_marqo_document(vespa_doc)
        self.assertEqual({
            "_id": "123",
            "int_field1": 1,
            "int_field2": 2,
            "int_map1.a": 3,  # int map is flattened
            "int_map1.b": 4,
            "float_field1": 1.0,
            "float_map1.a": 2.0,  # float map is flattened
            "float_map1.b": 3.0,
            "bool_field1": True,
            "bool_field2": False,
            "tags": ["foo", "bar"],
            "markets": ["fr", "au"],
            "title": "some product",
            "desc": "some awesome product",
            "marqo__tensors": {"title": {"chunks": ["some product"], "embeddings": [[1.0, 2.0]]}},
            'multimodal_params': {'multi_modal': {'type': 'multimodal_combination',
                                                  'weights': {'desc': 0.5, 'title': 1.0}}},
        }, marqo_doc)

    def test_vespa_to_marqo_conversion_should_handle_combined_string_array_fields(self):
        vespa_doc = {
            "id": "index1::123",
            "fields": {
                "marqo__id": "123",
                "marqo__string_array": ["tags::foo", "tags::bar", "markets::fr", "markets::au"]
            }
        }

        marqo_doc = self.vespa_index_prior_to_216.to_marqo_document(vespa_doc)

        self.assertEqual({
            "_id": "123",
            "tags": ["foo", "bar"],
            "markets": ["fr", "au"]
        }, marqo_doc)

    @patch('marqo.core.semi_structured_vespa_index.semi_structured_document.generate_uuid_str')
    def test_marqo_to_vespa_conversion(self, mock_generate_uuid_str):
        mock_generate_uuid_str.return_value = 'uuid1234'

        marqo_doc = {
            '_id': '123',
            'int_field1': 1,
            'int_map1': {'a': 1, 'b': 2},
            'float_field1': 1.0,
            'float_map1': {'aa': 2.0, 'bb': 3.0},
            'bool_field1': True,
            'bool_field2': False,
            'tags': ['foo', 'bar'],
            'markets': ['fr', 'au'],
            'title': 'some product',
            'desc': 'some awesome product',
            'marqo__tensors': {'title': {'chunks': ['some product'], 'embeddings': [[1.0, 2.0]]}},
        }

        vespa_doc = self.vespa_index.to_vespa_document(marqo_doc)

        self.assertEqual({
            'id': '123',
            'fields': {
                'marqo__bool_fields': {'bool_field1': 1, 'bool_field2': 0},
                'marqo__chunks_title': ['some product'],
                'marqo__embeddings_title': {'0': [1.0, 2.0]},
                'marqo__field_types': {'bool_field1': 'bool',
                                       'bool_field2': 'bool',
                                       'desc': 'string',
                                       'float_field1': 'float',
                                       'float_map1': 'float_map_entry',
                                       'float_map1.aa': 'float_map_entry',
                                       'float_map1.bb': 'float_map_entry',
                                       'int_field1': 'int',
                                       'int_map1': 'int_map_entry',
                                       'int_map1.a': 'int_map_entry',
                                       'int_map1.b': 'int_map_entry',
                                       'markets': 'string_array',
                                       'tags': 'string_array',
                                       'title': 'tensor'},
                'marqo__float_fields': {'float_field1': 1.0,
                                        'float_map1.aa': 2.0,
                                        'float_map1.bb': 3.0},
                'marqo__id': '123',
                'marqo__int_fields': {'int_field1': 1,
                                      'int_map1.a': 1,
                                      'int_map1.b': 2},
                'marqo__lexical_desc': 'some awesome product',
                'marqo__lexical_title': 'some product',
                'marqo__score_modifiers': {'float_field1': 1.0,
                                           'float_map1.aa': 2.0,
                                           'float_map1.bb': 3.0,
                                           'int_field1': 1,
                                           'int_map1.a': 1,
                                           'int_map1.b': 2},
                'marqo__short_string_fields': {'desc': 'some awesome product',
                                               'title': 'some product'},
                'marqo__string_array_markets': ['fr', 'au'],
                'marqo__string_array_tags': ['foo', 'bar'],
                'marqo__vector_count': 1,
                'marqo__version_uuid': 'uuid1234'
            },
        }, vespa_doc)

    def test_marqo_to_vespa_conversion_combined_string_array_fields(self):
        """For index created by Marqo prior to 2.16, string array fields are combined to one"""
        marqo_doc = {
            '_id': '123',
            'int_field1': 1,
            'int_map1': {'a': 1, 'b': 2},
            'float_field1': 1.0,
            'float_map1': {'aa': 2.0, 'bb': 3.0},
            'bool_field1': True,
            'bool_field2': False,
            'tags': ['foo', 'bar'],
            'markets': ['fr', 'au'],
            'title': 'some product',
            'desc': 'some awesome product',
            'marqo__tensors': {'title': {'chunks': ['some product'], 'embeddings': [[1.0, 2.0]]}},
        }

        vespa_doc = self.vespa_index_prior_to_216.to_vespa_document(marqo_doc)

        self.assertEqual({
            'id': '123',
            'fields': {
                'marqo__bool_fields': {'bool_field1': 1, 'bool_field2': 0},
                'marqo__chunks_title': ['some product'],
                'marqo__embeddings_title': {'0': [1.0, 2.0]},
                'marqo__float_fields': {'float_field1': 1.0,
                                        'float_map1.aa': 2.0,
                                        'float_map1.bb': 3.0},
                'marqo__id': '123',
                'marqo__int_fields': {'int_field1': 1,
                                      'int_map1.a': 1,
                                      'int_map1.b': 2},
                'marqo__lexical_desc': 'some awesome product',
                'marqo__lexical_title': 'some product',
                'marqo__score_modifiers': {'float_field1': 1.0,
                                           'float_map1.aa': 2.0,
                                           'float_map1.bb': 3.0,
                                           'int_field1': 1,
                                           'int_map1.a': 1,
                                           'int_map1.b': 2},
                'marqo__short_string_fields': {'desc': 'some awesome product',
                                               'title': 'some product'},
                # string arrays are combined
                'marqo__string_array': ['tags::foo',
                                        'tags::bar',
                                        'markets::fr',
                                        'markets::au'],
                'marqo__vector_count': 1
            },
        }, vespa_doc)

    def test_combine_number_stats_empty_current_stats(self):
        """Test _combine_number_stats when current_stats is empty"""
        stats = {"count": 5, "sum": 100, "avg": 20.0, "min": 10, "max": 30}
        result = self.vespa_index._combine_number_stats({}, stats)
        self.assertEqual(stats, result)

    def test_combine_number_stats_complete_stats(self):
        """Test _combine_number_stats with complete statistics"""
        current_stats = {"count": 3, "sum": 60, "avg": 20.0, "min": 15, "max": 25}
        stats = {"count": 2, "sum": 40, "avg": 20.0, "min": 10, "max": 30}
        
        result = self.vespa_index._combine_number_stats(current_stats, stats)
        
        expected = {
            "count": 5,  # 3 + 2
            "sum": 100,  # 60 + 40
            "avg": 20.0,  # (20.0 * 3 + 20.0 * 2) / (3 + 2) = 100 / 5
            "min": 10,   # min(15, 10)
            "max": 30    # max(25, 30)
        }
        self.assertEqual(expected, result)

    def test_combine_number_stats_partial_stats(self):
        """Test _combine_number_stats with partial statistics (missing some fields)"""
        current_stats = {"count": 4, "sum": 80, "min": 5}
        stats = {"count": 3, "max": 50}
        
        result = self.vespa_index._combine_number_stats(current_stats, stats)
        
        expected = {
            "count": 7,  # 4 + 3
            # sum not in both, so not included in result
            # avg not in both, so not included in result  
            # min only in current_stats, so not included
            # max only in stats, so not included
        }
        self.assertEqual(expected, result)

    def test_combine_number_stats_weighted_average_calculation(self):
        """Test _combine_number_stats weighted average calculation with different counts"""
        current_stats = {"count": 10, "avg": 15.0}
        stats = {"count": 5, "avg": 30.0}
        
        result = self.vespa_index._combine_number_stats(current_stats, stats)
        
        expected = {
            "count": 15,  # 10 + 5
            "avg": 20.0   # (15.0 * 10 + 30.0 * 5) / (10 + 5) = 300 / 15 = 20.0
        }
        self.assertEqual(expected, result)

    def test_combine_number_stats_min_max_edge_cases(self):
        """Test _combine_number_stats min/max with edge case values"""
        current_stats = {"count": 2, "min": -100, "max": 0}
        stats = {"count": 3, "min": 50, "max": -10}
        
        result = self.vespa_index._combine_number_stats(current_stats, stats)
        
        expected = {
            "count": 5,    # 2 + 3
            "min": -100,   # min(-100, 50)
            "max": 0       # max(0, -10)
        }
        self.assertEqual(expected, result)
