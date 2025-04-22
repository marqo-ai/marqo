import unittest

from marqo.tensor_search.tensor_search import select_attributes


class TestUnstructuredIndexSelectAttributes(unittest.TestCase):

    def test_return_meta_fields_when_attributes_to_retrieve_is_empty(self):
        """test that ["_id", "_score", "_highlights"] will be returned when attributes_to_retrieve is empty"""
        marqo_doc = {
            '_id': '123',
            '_score': 0.12,
            '_highlights': [{'title': 'Stay Elevated Dress Shoes - Black'}],
            'some_other_fields': 'hello',
            '_lexical_score': 0.11,  # FIXME: this is a meta field for hybrid search, but isn't returned
            '_tensor_score': 0.12,   # FIXME: this is a meta field for hybrid search, but isn't returned
        }

        expected_result = marqo_doc.copy()
        del expected_result['some_other_fields']
        del expected_result['_lexical_score']
        del expected_result['_tensor_score']

        filtered_doc = select_attributes(marqo_doc, {"_id", "_score", "_highlights"})
        self.assertEqual(expected_result, filtered_doc)

    def test_return_selected_field_and_meta_fields_when_attributes_to_retrieve_is_not_empty(self):
        marqo_doc = {
            '_id': '123',
            '_score': 0.12,
            'string_field': 'aaa',
            'int_field': 123,
            'float_field': 1.23,
            'bool_field': True,
            'string_array_field': ['a', 'b'],
            'some_other_fields': 'hello',
        }

        expected_result = marqo_doc.copy()
        del expected_result['some_other_fields']

        attributes_to_retrieve = {'string_field', 'int_field', 'float_field', 'bool_field', 'string_array_field',
                                  "_id", "_score", "_highlights", "field_not_exist_in_doc"}

        filtered_doc = select_attributes(marqo_doc, attributes_to_retrieve)
        self.assertEqual(expected_result, filtered_doc)

    def test_return_selected_map_field_flattened(self):
        marqo_doc = {
            '_id': '123',
            '_score': 0.12,
            'int_map_field1.a': 1,
            'int_map_field1.b': 2,
            'int_map_field2.c': 3,
            'int_map_field2.d': 4,
            'float_map_field1.a': 1.0,
            'float_map_field1.b': 2.0,
            'float_map_field2.c': 3.0,
            'float_map_field2.d': 4.0,
            'some_other_fields': 'hello',
        }

        expected_result = marqo_doc.copy()
        del expected_result['some_other_fields']
        del expected_result['int_map_field2.c']
        del expected_result['int_map_field2.d']
        del expected_result['float_map_field1.a']
        del expected_result['float_map_field1.b']

        attributes_to_retrieve = {'int_map_field1', 'float_map_field2', "_id", "_score", "_highlights"}

        filtered_doc = select_attributes(marqo_doc, attributes_to_retrieve)
        self.assertEqual(expected_result, filtered_doc)
