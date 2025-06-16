from unittest.mock import patch

from marqo.core.models import MarqoQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
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
