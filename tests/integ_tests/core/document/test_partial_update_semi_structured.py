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

        semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured_12')
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
            "string_array": ["aaa", "bbb"],
            "string_array2": ["123", "456"],
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
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        for doc in [self.doc, self.doc2, self.doc3]:
            id = doc['_id']
            res = self.config.document.partial_update_documents([{'_id': id, 'bool_field': False}], self.index)
            self.assertFalse(res.errors)

        for doc in [self.doc, self.doc2, self.doc3]:
            id = doc['_id']
            doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
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

    def test_partial_update_should_not_non_existent_field(self): #This now works
        res = self.config.document.partial_update_documents([{'_id': '2', 'update_field_that_doesnt_exist': 500}], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(500, doc['update_field_that_doesnt_exist'])
        self._assert_fields_unchanged(doc, ['update_field_that_doesnt_exist'])

    def test_partial_update_should_not_update_int_field_to_float(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': 1.0}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_update_float_field_to_float(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(500.0, doc['float_field'])
        self._assert_fields_unchanged(doc, ['float_field'])
        pass

    def test_partial_update_should_update_int_map(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {'a': 2, 'b': 3}}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['int_map.a'], 2)
        self.assertEqual(doc['int_map.b'], 3)
        self._assert_fields_unchanged(doc, ['int_map'])


    def test_partial_update_should_update_float_map(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'float_map': {'c': 2.0, 'd': 3.0}}],
                                                            self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['float_map.c'], 2.0)
        self.assertEqual(doc['float_map.d'], 3.0)
        self._assert_fields_unchanged(doc, ['float_map'])


    def test_partial_update_should_allow_changing_numeric_types_in_map(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {
            'a': 2,  # update int to int
        }, 'float_map': {
            'c': 3.0,  # update float to int
        }}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['int_map.a'], 2)
        self.assertEqual(doc['float_map.c'], 3.0)

    def test_partial_update_should_update_string_array(self):

        res = self.config.document.partial_update_documents([{'_id': '2', 'string_array': ["ccc"]}], self.config.index_management.get_index(self.index.name))
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(["ccc"], doc['string_array'])
        self._assert_fields_unchanged(doc, ['string_array'])

    def test_partial_update_should_update_short_string(self):
        index = self.config.index_management.get_index(self.index.name)
        res = self.config.document.partial_update_documents(
            [{'_id': '2', 'short_string_field': 'updated_short_string'}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('updated_short_string', doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

    def test_partial_update_should_update_long_string(self):
        index = self.config.index_management.get_index(self.index.name)
        res = self.config.document.partial_update_documents(
            [{'_id': '2', 'long_string_field': 'updated_long_string' * 10}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('updated_long_string' * 10, doc['long_string_field'])
        self._assert_fields_unchanged(doc, ['long_string_field'])

    def test_partial_update_should_update_long_string_to_short_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'long_string_field:{self.doc2["long_string_field"]}') #Note: Not all testing documents defined above contain short_string_field key, so when you change to run all the tests by running a for loop make sure to take a look here
        self.assertEqual(0, len(res['hits']))
        index = self.config.index_management.get_index(self.index.name)

        res = self.config.document.partial_update_documents([{'_id': '2', 'long_string_field': 'short'}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('short', doc['long_string_field'])
        self._assert_fields_unchanged(doc, ['long_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*', filter=f'long_string_field:short')
        self.assertEqual(1, len(res['hits']))

    def test_partial_update_should_update_short_string_to_long_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{self.doc2["short_string_field"]}') #Note: Not all testing documents defined above contain short_string_field key, so when you change to run all the tests by running a for loop make sure to take a look here
        self.assertEqual(2, len(res['hits']))

        index = self.config.index_management.get_index(self.index.name)

        res = self.config.document.partial_update_documents([{'_id': '2', 'short_string_field': 'verylongstring'*10}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('verylongstring'*10, doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{doc["short_string_field"]}')
        self.assertEqual(0, len(res['hits']))

    def test_partial_update_should_update_score_modifiers(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {
            'a': 2,  # update int to int
        }, 'float_map': {
            'c': 3.0,  # update float to int
        }}], self.index)
        self.assertFalse(res.errors)
        res = self.config.vespa_client.get_document('2', self.config.index_management.get_index(self.index.name).schema_name)
        doc = res.document.dict().get('fields')
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_field'], 123.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_field'], 123.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.a'], 2.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.b'], 2.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], 3.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.d'], 2.0)


    # Test update multiple fields
    def test_partial_update_should_update_multiple_fields(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': 500, 'bool_field': False, 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')

    def test_partial_update_should_add_new_fields(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'new_field': 500}], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(500, doc['new_field'])

    # Reject any tensor field change
    def test_partial_update_should_reject_tensor_field(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'tensor_field': 'new_title'}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_tensor_subfield(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'tensor_subfield': 'new_description'}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_custom_vector_field(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'custom_vector_field': {
            "content": "efgh",
            "vector": [1.0] * 32
        }}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_multimodal_combo_field(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'multimodal_combo_field': {
            "tensor_field": "new_title",
            "tensor_subfield": "new_description"
        }}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_numeric_array_field_type(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_array': [1, 2, 3]}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_should_reject_new_lexical_field(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'new_lexical_field': 'some string that signifies new lexical field'}], self.index)
        self.assertTrue(res.errors)

    def test_partial_update_invalid_field_name(self):
        with pytest.raises(InvalidFieldNameError):
            res = self.config.document.partial_update_documents([{'_id': '2', 'marqo__': 1}], self.index)
            self.assertTrue(res.errors)

    def test_partial_update_sort_of_backwards_compatibility_test(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'string_array': ["ccc"]}], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')