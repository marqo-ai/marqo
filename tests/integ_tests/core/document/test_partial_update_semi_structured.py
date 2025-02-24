from typing import List, Dict, Any

import pytest

from marqo.api.exceptions import InvalidFieldNameError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from integ_tests.marqo_test import MarqoTestCase

class TestPartialUpdate(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured_13')
        cls.create_indexes([semi_structured_index_request])
        cls.index = cls.indexes[0]

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

    def test_update_numeric_array_field(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'numeric_array1': [4, 5]}], self.config.index_management.get_index(self.index.name))
        self.assertTrue(res.errors)

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

    def test_partial_update_should_update_int_map_with_new_value(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {
            'd': 2
          }
        }], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['int_map.d'], 2)


    def test_partial_update_should_update_float_map(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'float_map': {'c': 2.0, 'd': 3.0}}],
                                                            self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['float_map.c'], 2.0)
        self.assertEqual(doc['float_map.d'], 3.0)
        self._assert_fields_unchanged(doc, ['float_map'])


    def test_partial_update_should_allow_changing_numeric_types_in_map(self):
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': 2, 'int_map': {
            'a': 2,  # update int to int
        }, 'float_map': {
            'c': 3.0,  # update float to float
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
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], 3.0)


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

    def test_partial_update_trying_to_add_multi_modal_field(self):
        res = self.config.document.partial_update_documents([{'_id': '1',
            "tensor_subfield": "new_title",
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('tensor_subfield of type str does not exist in the original document. We do not support adding new lexical fields in partial updates', res.items[0].error)

    def test_partial_update_should_handle_mixed_numeric_map_updates(self):
        """Test updating maps with mix of additions and removals"""
        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_map': {
                'a': 10,  # Update existing
                'c': 3,   # Add new
                'b': 20   # Update existing
            },
            'float_map': {
                'c': 10.5,  # Update existing
                'e': 5.5    # Add new
            }
        }], self.index)
        self.assertFalse(res.errors)
        
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(10, doc['int_map.a'])
        self.assertEqual(20, doc['int_map.b'])
        self.assertEqual(3, doc['int_map.c'])
        self.assertEqual(10.5, doc['float_map.c'])
        self.assertEqual(5.5, doc['float_map.e'])

    def test_partial_update_should_reject_invalid_map_values(self):
        """Test rejection of invalid value types in numeric maps"""
        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_map': {
                'a': 'string',  # Invalid type
                'b': 2.5,      # Invalid type
                'c': True      # Invalid type
            }
        }], self.index)
        self.assertTrue(res.errors)
        
        # Verify original values unchanged
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(1, doc['int_map.a'])
        self.assertEqual(2, doc['int_map.b'])

    def test_partial_update_should_handle_multiple_docs(self):
        """Test updating multiple documents in one request"""
        updates = [
            {
                '_id': '2',
                'int_field': 1000,
                'float_map': {'c': 99.9}
            },
            {
                '_id': '3', 
                'bool_field': False,
                'int_map': {'a': 777}
            }
        ]
        res = self.config.document.partial_update_documents(updates, self.index)
        self.assertFalse(res.errors)
        
        # Verify updates
        doc2 = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(1000, doc2['int_field'])
        self.assertEqual(99.9, doc2['float_map.c'])
        
        doc3 = tensor_search.get_document_by_id(self.config, self.index.name, '3')
        self.assertFalse(doc3['bool_field'])
        self.assertEqual(777, doc3['int_map.a'])

    def test_partial_update_should_handle_duplicate_doc_ids(self):
        """Test handling of duplicate document IDs in update request"""
        updates = [
            {
                '_id': '2',
                'int_field': 100
            },
            {
                '_id': '2',
                'int_field': 200
            }
        ]
        res = self.config.document.partial_update_documents(updates, self.index)
        self.assertFalse(res.errors)
        
        # Verify last update wins
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(200, doc['int_field'])

    def test_partial_update_should_handle_non_existent_doc_id(self):
        """Test updating non-existent document"""
        res = self.config.document.partial_update_documents([{
            '_id': 'non_existent',
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('marqo vector store either cannot find the document you are trying to update, or you are trying to change type of a variable as part of an update request which is not allowed. please fix the request and try again', res.items[0].error.lower())

    def test_partial_update_should_handle_none_id(self):
        """Test handling of None _id field"""
        res = self.config.document.partial_update_documents([{
            '_id': None,
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('_id', res.items[0].error.lower())

    def test_partial_update_should_handle_missing_id(self):
        """Test handling of document without _id field"""
        res = self.config.document.partial_update_documents([{
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('_id', res.items[0].error.lower())

    def test_partial_update_should_handle_empty_update_list(self):
        """Test handling of empty document list"""
        res = self.config.document.partial_update_documents([], self.index)
        self.assertFalse(res.errors)
        self.assertEqual(0, len(res.items))

    def test_partial_update_should_handle_mixed_valid_invalid_docs(self):
        """Test batch with mix of valid and invalid documents"""
        updates = [
            {
                '_id': '2',
                'int_field': 100
            },
            {
                '_id': '3',
                'bool_field': True
            },
            {
                'missing_id': True
            }
        ]
        res = self.config.document.partial_update_documents(updates, self.index)
        self.assertTrue(res.errors)

        # Verify valid updates succeeded
        doc2 = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(100, doc2['int_field'])

        doc3 = tensor_search.get_document_by_id(self.config, self.index.name, '3')
        self.assertTrue(doc3['bool_field'])

        # Verify error responses for invalid docs
        self.assertEqual(3, len(res.items))
        print(res)
        self.assertFalse(res.items[0].error)  # Valid doc
        self.assertFalse(res.items[1].error)  # Valid doc
        self.assertIn("'_id' is a required field", res.items[2].error)  # Missing ID

    def test_partial_update_should_handle_nested_maps(self):
        """Test handling of nested maps in updates"""
        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_map': {
                'nested': {
                    'too': 'deep'
                }
            }
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('unsupported field type', res.items[0].error.lower())

    def test_partial_update_should_preserve_other_fields(self):
        """Test that non-updated fields remain unchanged"""
        original_doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')

        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_field': 999
        }], self.index)
        self.assertFalse(res.errors)

        updated_doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')

        # Verify updated field
        self.assertEqual(999, updated_doc['int_field'])

        # Verify all other fields unchanged
        for field, value in original_doc.items():
            if field != 'int_field':
                self.assertEqual(value, updated_doc.get(field),
                                 f"Field {field} changed unexpectedly")

    def test_partial_update_should_handle_empty_string_id(self):
        """Test handling of empty string as document ID"""
        res = self.config.document.partial_update_documents([{
            '_id': '',
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn("document id can't be empty", res.items[0].error.lower())
