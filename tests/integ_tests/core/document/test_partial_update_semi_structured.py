from typing import List, Dict, Any

import pytest

from marqo.api.exceptions import InvalidFieldNameError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from tests.integ_tests.marqo_test import MarqoTestCase


class TestPartialUpdate(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured_10')
        cls.create_indexes([semi_structured_index_request])
        cls.index = cls.indexes[0]
        # cls.index = cls.config.index_management.get_index('test_partial_update_semi_structured')

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.doc = {
            '_id': '1',
            # 'tensor_field': 'title',
            # 'tensor_subfield': 'description',
            # "short_string_field": "shortstring",
            # "long_string_field": "Thisisaverylongstring" * 10,
            # "int_field": 123,
            # "float_field": 123.0,
            "string_array": ["aaa", "bbb"],
            "string_array2": ["123", "456"],
            # "int_map": {"a": 1, "b": 2},
            # "float_map": {"c": 1.0, "d": 2.0},
            # "bool_field": True,
            # "bool_field2": False,
            # "custom_vector_field": {
            #     "content": "abcd",
            #     "vector": [1.0] * 32
            # }
        }
        self.doc2 = {
            '_id': '2',
            'tensor_field': 'title',
            'tensor_subfield': 'description',
            "short_string_field": "shortstring",
            "long_string_field": "Thisisaverylongstring" * 10,
            "int_field": 123,
            "float_field": 123.0,
            "string_array": ["aaa", "bbb"],
            "string_array2": ["123", "456"],
            "int_map": {"a": 1, "b": 2},
            "float_map": {"c": 1.0, "d": 2.0},
            "bool_field": True,
            "bool_field2": False,
            "custom_vector_field": {
                "content": "abcd",
                "vector": [1.0] * 32
            }
        }
        self.doc3 = {
            '_id': '3',
            'tensor_field': 'title',
            'tensor_subfield': 'description',
            "short_string_field": "shortstring",
            "long_string_field": "Thisisaverylongstring" * 10,
            "int_field": 123,
            "float_field": 123.0,
            "int_map": {"a": 1, "b": 2},
            "float_map": {"c": 1.0, "d": 2.0},
            "bool_field": True,
            "bool_field2": False,
            "custom_vector_field": {
                "content": "abcd",
                "vector": [1.0] * 32
            }
        }
        self.add_documents(self.config, add_docs_params=AddDocsParams(
            index_name=self.index.name,
            docs=[self.doc, self.doc2, self.doc3],
            tensor_fields=['tensor_field', 'custom_vector_field', 'multimodal_combo_field'],
            mappings = {
                "custom_vector_field": {"type": "custom_vector"},
                "multimodal_combo_field": {
                    "type": "multimodal_combination",
                    "weights": {"tensor_field": 1.0, "tensor_subfield": 2.0}
                }
            }
        ))

    def _assert_fields_unchanged(self, doc: Dict[str, Any], excluded_fields: List[str]):
        for field, value in doc.items():
            if field in excluded_fields:
                continue
            elif field == 'custom_vector_field':
                continue
            elif isinstance(value, dict):
                for k, v in value.items():
                    flattened_field_name = f'{field}.{k}'
                    self.assertEqual(v, doc.get(flattened_field_name, None), f'{flattened_field_name} is changed.')
            else:
                self.assertEqual(value, doc.get(field, None), f'{field} is changed.')

    # Test update single field
    def test_partial_update_should_update_bool_field(self):
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print("Printing the document", doc)
        for doc in [self.doc, self.doc2, self.doc3]:
            id = doc['_id']
            res = self.config.document.partial_update_documents([{'_id': id, 'bool_field': False}], self.index)
            print("Printing partial docs update", res)
            self.assertFalse(res.errors)

        for doc in [self.doc, self.doc2, self.doc3]:
            id = doc['_id']
            doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
            print("Printing get docs response", doc)
            self.assertFalse(doc['bool_field'])
            self._assert_fields_unchanged(doc, ['bool_field'])

    def test_partial_update_should_update_int_field_to_int(self):
        for doc in [self.doc, self.doc2, self.doc3]:
            res = self.config.document.partial_update_documents([{'_id': doc['_id'], 'int_field': 500}], self.index)
            self.assertFalse(res.errors)

        for doc in [self.doc, self.doc2, self.doc3]:
            id = doc['_id']
            doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
            self.assertEqual(500, doc['int_field'])
            self._assert_fields_unchanged(doc, ['int_field'])

    #TODO: I Haven't implemented that thing where the metadata map will also be updated for this field. something like update_field_that_doesnt_exist <-> int
    def test_partial_update_should_not_non_existent_field(self): #This now works
        res = self.config.document.partial_update_documents([{'_id': '1', 'update_field_that_doesnt_exist': 500}], self.index)
        # self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)
        self.assertEqual(500, doc['update_field_that_doesnt_exist'])
        self._assert_fields_unchanged(doc, ['update_field_that_doesnt_exist'])

        #TODO: This shouldn't work - don't implement it

    def test_partial_update_should_not_update_int_field_to_float(self):
        # So something like - let it update if the request says it's int - let it update if it is either int or a float.
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_field': 1.0}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_update_float_field_to_float(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)
        self.assertEqual(500.0, doc['float_field'])
        self._assert_fields_unchanged(doc, ['float_field'])
        pass

    def test_partial_update_should_update_int_map(self):
        # res = self.config.document.partial_update_documents([{'_id': '1', 'int_map': {}])
        pass

    def test_partial_update_should_update_float_map(self):
        pass

    # @pytest.mark.skip(reason="This feature is not implemented yet.")
    # Details: This is only update - purely only update, no removal or addition happens. If you pass int_map : {'a': 2} it will only update a and not change other values in the int_map
    def test_partial_update_should_allow_changing_numeric_types_in_map(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_map': {
            'a': 2,  # update int to int
            # 'a': 2.0,  # TODO: update int to float THis shouldn't work anyway.
            # 'c': 3,  # add new int value #TODO: This shouldn't work anyway - because it will look for this field's type in the metadata and won't find it so pre-condition will fail.
            # 'd': 4.0  # add new float value #TODO: This shouldn't work either.
        }, 'float_map': {
            'c': 3.0,  # update float to int #TODO: This should work.
        }}], self.index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)
        self.assertEqual(2.0, doc['int_map.a'])
        # self.assertNotIn('int_map.b', doc)  # [Update: No it cannot be deleted because since you don't fetch the docuemnt - you don't know whether a field has been added or removed, the best guess is that field has been updated so we create a assign statement as opposed to remove statement etc) b will be deleted, as it's not in the partial_update_documents call. This seems like we make updates at a field level, not at things defined in the field. So if I wanna update int_map entire int_map will be updated together, I cannot go and change int_map.get('a') to something else just. I will have to specific int_map {'a': 2.0, 'b': 3} such that b is not deleted in the process.
        self.assertEqual(3.0, doc['float_map.c'])
        # self.assertEqual(4.0, doc['int_map.d'])
        # self._assert_fields_unchanged(doc, ['int_map'])

    #note: This scenario will not work. I mean if the original document has something like {marqo__string_array:
    # [ 'string_array::aaa', 'string_array:bbb', 'string_array2::123', 'string_array2::456' ]} and you try to update it to
    # {marqo__string_array: [ 'string_array::ccc' ]} it will not work. It will only work if you update the entire field.
    # So if you update the entire field to {marqo__string_array: [ 'string_array::ccc' ]} it will not work - as in it will
    # just change all of it to {marqo__string_array: [ 'string_array::ccc' ]}, thus losing the information stored under string_array2::123, or string_array2::456.
    # This is because you can just update the entire Marqo__string_array field together, you cannot update individual elements of the array.
    def test_partial_update_should_update_string_array(self):
        # doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        # print(doc)
        res = self.config.document.partial_update_documents([{'_id': '1', 'string_array': ["ccc"]}], self.index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)
        self.assertEqual(["ccc"], doc['string_array'])
        self._assert_fields_unchanged(doc, ['string_array'])

    def test_partial_update_should_update_short_string(self):
        pass

    def test_partial_update_should_update_long_string(self):
        pass

    #TODO: looks like this is Unimplemented -
    def test_partial_update_should_update_long_string_to_short_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'long_string_field:{self.doc["long_string_field"]}')
        self.assertEqual(0, len(res['hits']))
        print(res)
        # note: Isme wapas index leke aana padd raha hai - earlier in Yihan's test case we just use to pass self.config.document.partial_udpate_docs(..., self.index) - somehow the self.index value is not getting updated here.
        # note: But what is curious is that when I switch to Yihan's branch - it works. The test case is almost the same over there.
        index = self.config.index_management.get_index(self.index.name)

        res = self.config.document.partial_update_documents([{'_id': '1', 'long_string_field': 'short'}], index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual('short', doc['long_string_field'])
        self._assert_fields_unchanged(doc, ['long_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*', filter=f'long_string_field:short')
        self.assertEqual(1, len(res['hits']))

    def test_partial_update_should_update_short_string_to_long_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{self.doc["short_string_field"]}')
        self.assertEqual(1, len(res['hits']))
        # note: Isme wapas index leke aana padd raha hai - earlier in Yihan's test case we just use to pass self.config.document.partial_udpate_docs(..., self.index) - somehow the self.index value is not getting updated here.
        # note: But what is curious is that when I switch to Yihan's branch - it works. The test case is almost the same over there.

        index = self.config.index_management.get_index(self.index.name)

        res = self.config.document.partial_update_documents([{'_id': '1', 'short_string_field': 'verylongstring'*10}], index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual('verylongstring'*10, doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{doc["short_string_field"]}')
        self.assertEqual(0, len(res['hits']))

    def test_partial_update_should_update_score_modifiers(self):

        pass

    # Test update multiple fields
    def test_partial_update_should_update_multiple_fields(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_field': 500, 'bool_field': False, 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)

    # Test remove field
    # This feature itself is unimplemented. #TODO: This is not possible if we go with the metadata map approach
    @pytest.mark.skip(reason = "This is not possible if we go with the metadata map approach")
    def test_partial_update_should_remove_field_if_set_to_none(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_field': None}], self.index)
        print(res)
        # self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        print(doc)
        # self.assertNotIn('int_field', doc)

    # Test add new fields
    # @pytest.mark.skip(reason="This feature is not implemented yet.")
    # Note: this is already implemented somewhere above
    def test_partial_update_should_add_new_fields(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'new_field': 500}], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual(500, doc['new_field'])

    # Reject any tensor field change
    def test_partial_update_should_reject_tensor_field(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'tensor_field': 'new_title'}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_tensor_subfield(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'tensor_subfield': 'new_description'}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_custom_vector_field(self): #TODO: This isn't working for some reason - need to debug
        res = self.config.document.partial_update_documents([{'_id': '1', 'custom_vector_field': {
            "content": "efgh",
            "vector": [1.0] * 32
        }}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_multimodal_combo_field(self): #TODO: Not working - need to debug
        res = self.config.document.partial_update_documents([{'_id': '1', 'multimodal_combo_field': {
            "tensor_field": "new_title",
            "tensor_subfield": "new_description"
        }}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_numeric_array_field_type(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_array': [1, 2, 3]}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_new_lexical_field(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'new_lexical_field': 'some string that signifies new lexical field'}], self.index)
        print(res)
        self.assertTrue(res.errors)

    def test_partial_update_invalid_field_name(self):
        with pytest.raises(InvalidFieldNameError):
            res = self.config.document.partial_update_documents([{'_id': '1', 'marqo__': 1}], self.index)
            print(res)
            self.assertTrue(res.errors)

    def test_partial_update_sort_of_backwards_compatibility_test(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'string_array': ["ccc"]}], self.index)
        print(res)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')