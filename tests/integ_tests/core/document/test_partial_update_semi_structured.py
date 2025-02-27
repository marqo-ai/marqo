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

        semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured_14')
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
            },
            "lexical_field": "some string that signifies lexical field"
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
        self.id_to_doc = {
            '1': self.doc,
            '2': self.doc2,
            '3': self.doc3
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
        self.index = self.config.index_management.get_index(self.index.name)

    def _assert_fields_unchanged(self, doc: Dict[str, Any], excluded_fields: List[str]):
        """Verify that fields in the document remain unchanged except for the specified excluded fields.
        
        This helper method checks that all fields in the document match their expected values,
        excluding the fields that were intentionally modified during the test.
        
        Args:
            doc: The document to check
            excluded_fields: List of field names that were intentionally modified and should be excluded from verification
        """
        doc_id = doc['_id']
        doc_to_compare_against = self.id_to_doc[doc_id]
        for field, value in doc_to_compare_against.items():
            if field in excluded_fields:
                continue
            elif field == 'custom_vector_field':
                self.assertEqual(value['content'], doc.get(field, None), f'{field} is changed.')
            elif isinstance(value, dict):
                for k, v in value.items():
                    flattened_field_name = f'{field}.{k}'
                    if flattened_field_name in excluded_fields:
                        continue
                    self.assertEqual(v, doc.get(flattened_field_name, None), f'{flattened_field_name} is changed.')
            else:
                self.assertEqual(value, doc.get(field, None), f'{field} is changed.')

    # Test update single field
    def test_partial_update_should_update_bool_field(self):
        """Test that boolean fields can be updated correctly via partial updates.
        
        This test verifies that boolean fields can be updated for multiple documents
        while ensuring other fields remain unchanged.
        """
        test_docs = [self.doc, self.doc2]
        
        # First update the documents
        for doc in test_docs:
            with self.subTest(f"Updating document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'bool_field': False}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")
        
        # Then verify the updates
        for doc in test_docs:
            with self.subTest(f"Verifying document with ID {doc['_id']}"):
                id = doc['_id']
                updated_doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertFalse(updated_doc['bool_field'], f"Expected bool_field to be False for document {id}")
                self._assert_fields_unchanged(updated_doc, ['bool_field'])

    def test_partial_update_should_update_int_field_to_int(self):
        """Test that integer fields can be updated correctly via partial updates.
        
        This test verifies that integer fields can be updated for multiple documents
        while ensuring other fields remain unchanged.
        """
        test_docs = [self.doc, self.doc2, self.doc3]
        
        # First update the documents
        for doc in test_docs:
            with self.subTest(f"Updating document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_field': 500}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")
        
        # Then verify the updates
        for doc in test_docs:
            with self.subTest(f"Verifying document with ID {doc['_id']}"):
                id = doc['_id']
                updated_doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(500, updated_doc['int_field'], f"Expected int_field to be 500 for document {id}")
                self._assert_fields_unchanged(updated_doc, ['int_field'])

    def test_partial_update_to_non_existent_field(self): 
        """Test that partial updates to non-existent fields are successful.
        
        This test case basically verifies that we can add new fields via partial updates
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'update_field_that_doesnt_exist': 500}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(500, doc['update_field_that_doesnt_exist'])
        self._assert_fields_unchanged(doc, ['update_field_that_doesnt_exist'])

    def test_partial_update_should_not_update_int_field_to_float(self):
        """Test that partial updates to int fields are rejected when the value is a float.
        
        This test verifies that partial updates to int fields are rejected when the value is a float.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': 1.0}], self.index)
        self.assertTrue(res.errors)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_update_float_field_to_float(self):
        """Test that partial updates to float fields are successful.
        
        This test verifies that partial updates to float fields are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(500.0, doc['float_field'])
        self._assert_fields_unchanged(doc, ['float_field'])

    def test_partial_update_should_update_int_map(self):
        """Test that partial updates to int maps are successful.
        
        This test verifies that partial updates to int maps are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {'a': 2, 'b': 3}}], self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['int_map.a'], 2)
        self.assertEqual(doc['int_map.b'], 3)
        self._assert_fields_unchanged(doc, ['int_map'])

    def test_partial_update_should_update_int_map_with_new_value(self):
        """Test that partial updates to int maps with new values are successful.
        
        This test verifies that partial updates to int maps with new values are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {
            'd': 2
          }
        }], self.index)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertIsNone(doc.get('int_map.a'))
        self.assertIsNone(doc.get('int_map.b'))
        self.assertEqual(doc['int_map.d'], 2)


    def test_partial_update_should_update_float_map(self):
        """Test that partial updates to float maps are successful.
        
        This test verifies that partial updates to float maps are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'float_map': {'c': 2.0, 'd': 3.0}}],
                                                            self.index)
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(doc['float_map.c'], 2.0)
        self.assertEqual(doc['float_map.d'], 3.0)
        self._assert_fields_unchanged(doc, ['float_map'])


    def test_partial_update_should_allow_changing_multiple_maps_in_same_request(self):
        """Test that partial updates to multiple maps in the same request are successful.
        
        This test verifies that partial updates to multiple maps in the same request are successful.
        """


        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': 2, 'int_map': {
            'a': 2,  # update int to int
        }, 'float_map': {
            'c': 3.0,  # update float to float
        }, 'bool_field': False, 'float_field': 500.0}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(2, doc['int_field'])
        self.assertFalse(doc['bool_field'])
        self.assertEqual(500.0, doc['float_field'])
        self.assertEqual(doc['int_map.a'], 2)
        self.assertIsNone(doc.get('int_map.b'))
        self.assertEqual(doc['float_map.c'], 3.0)
        self.assertIsNone(doc.get('float_map.d'))
        self._assert_fields_unchanged(doc, ['int_map.a', 'int_map.b', 'float_map.d', 'float_map.c', 'int_field', 'bool_field', 'float_field'])

    def test_partial_update_should_update_string_array(self):
        """Test that partial updates to string arrays are successful.
        
        This test verifies that partial updates to string arrays are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'string_array': ["ccc"]}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(["ccc"], doc['string_array'])
        self._assert_fields_unchanged(doc, ['string_array'])

    def test_partial_update_should_reject_new_string_array_field(self):
        """Test that partial updates to string arrays are successful.

        This test verifies that partial updates to string arrays are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'string_array3': ["ccc"]}],
                                                            self.index)
        self.assertTrue(res.errors)
        self.assertEqual(400, res.items[0].status)
        self.assertIn('Unstructured index updates only support updating existing string array fields', res.items[0].error)

    def test_partial_update_should_allow_adding_new_string_string_array_field_if_present_in_other_docs_in_same_index(self):
        """Tests that partial updates allow adding new string / string array fields if they are present in some other document in the same index.

        For example, doc2 contains lexical_field and string_array. Hence when we try to add lexical_field and string_array to doc1, it should be allowed.
        """
        res = self.config.document.partial_update_documents([{'_id': '1', "lexical_field": "some value 2", 'string_array': ["ccc"]}],
                                                            self.config.index_management.get_index(self.index.name))
        self.assertFalse(res.errors)
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual("some value 2", doc['lexical_field'])
        self.assertEqual(["ccc"], doc['string_array'])

    def test_partial_update_should_update_short_string(self):
        """Test that partial updates to short strings are successful.
        
        This test verifies that partial updates to short strings are successful.
        """
        index = self.config.index_management.get_index(self.index.name)
        res = self.config.document.partial_update_documents(
            [{'_id': '2', 'short_string_field': 'updated_short_string'}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('updated_short_string', doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

    def test_partial_update_should_update_long_string(self):
        """Test that partial updates to long strings are successful.
        
        This test verifies that partial updates to long strings are successful.
        """
        index = self.config.index_management.get_index(self.index.name)
        res = self.config.document.partial_update_documents(
            [{'_id': '2', 'long_string_field': 'updated_long_string' * 10}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('updated_long_string' * 10, doc['long_string_field'])
        self._assert_fields_unchanged(doc, ['long_string_field'])

    def test_partial_update_should_update_long_string_to_short_string(self):
        """Test that partial updates to long strings are successful.
        
        This test verifies that partial updates to long strings are successful.
        """
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'long_string_field:{self.doc2["long_string_field"]}')
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
        """Test that partial updates to short strings are successful.
        
        This test verifies that partial updates to short strings are successful.
        """
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{self.doc2["short_string_field"]}')
        self.assertEqual(2, len(res['hits']))

        index = self.config.index_management.get_index(self.index.name)

        res = self.config.document.partial_update_documents([{'_id': '2', 'short_string_field': 'verylongstring'*10}], index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual('verylongstring'*10, doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{self.doc3["short_string_field"]}')
        self.assertEqual(1, len(res['hits']))

    def test_partial_update_should_update_score_modifiers(self):
        """Test that partial updates to score modifiers are successful.
        
        This test verifies that partial updates to score modifiers are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': {
            'a': 2,  # update int to int
            'd': 5,  # new entry in int map
        }, 'float_map': {
            'c': 3.0,  # update float to float
        }, 'new_int' : 1,  # new int field
            'new_float': 2.0,  # new float field
            'new_map': {'a': 1, 'b': 2.0},  # new map field
          }], self.index)
        self.assertFalse(res.errors)
        res = self.config.vespa_client.get_document('2', self.config.index_management.get_index(self.index.name).schema_name)
        doc = res.document.dict().get('fields')
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_field'], 123.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_field'], 123.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.a'], 2.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], 3.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells'].get('int_map.b', None), None)
        self.assertEqual(doc['marqo__score_modifiers']['cells'].get('float_map.d', None), None)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['new_int'], 1.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['new_float'], 2.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['new_map.a'], 1.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['new_map.b'], 2.0)
        self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.d'], 5.0)


    def test_partial_update_should_add_new_fields(self):
        """Test that partial updates to new fields are successful.
        
        This test verifies that partial updates to new fields are successful.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'new_field': 500, 'new_float': 500.0,
                                                              'new_int_map':{'a':2},
                                                              'new_bool_field': True,
                                                              'new_float_field': 10.0
                                                              }], self.config.index_management.get_index(self.index.name))
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self._assert_fields_unchanged(doc, [])
        self.assertEqual(500.0, doc['new_float'])
        self.assertEqual(2, doc['new_int_map.a'])
        self.assertEqual(500, doc['new_field'])
        self.assertEqual(True, doc['new_bool_field'])
        self.assertEqual(10.0, doc['new_float_field'])

    # Reject any tensor field change
    def test_partial_update_should_reject_tensor_field(self):
        """Test that partial updates to tensor fields are rejected.
        
        This test verifies that partial updates to tensor fields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'tensor_field': 'new_title'}], self.index)
        self.assertTrue(res.errors)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_multi_modal_field_subfield(self):
        """Test that partial updates to tensor subfields are rejected.
        
        This test verifies that partial updates to tensor subfields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'tensor_subfield': 'new_description'}], self.index)
        self.assertTrue(res.errors)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_custom_vector_field(self):
        """Test that partial updates to custom vector fields are rejected.
        
        This test verifies that partial updates to custom vector fields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'custom_vector_field': {
            "content": "efgh",
            "vector": [1.0] * 32
        }}], self.index)
        self.assertTrue(res.errors)
        self.assertEqual(400, res.items[0].status)
        self.assertIn("Unsupported field type <class 'str'> for field custom_vector_field in doc 2. "
                      "We only support int and float types for map values when updating a document", res.items[0].error)

    def test_partial_update_should_reject_multimodal_combo_field(self):
        """Test that partial updates to multimodal combo fields are rejected.
        
        This test verifies that partial updates to multimodal combo fields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'multimodal_combo_field': {
            "tensor_field": "new_title",
            "tensor_subfield": "new_description"
        }}], self.index)
        self.assertTrue(res.errors)
        self.assertIn("Unsupported field type <class 'str'> for field multimodal_combo_field in doc 2", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_numeric_array_field_type(self):
        """Test that partial updates to numeric array fields are rejected.
        
        This test verifies that partial updates to numeric array fields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_array': [1, 2, 3]}], self.index)
        self.assertTrue(res.errors)
        self.assertIn("Unstructured index updates only support updating existing string array fields", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_new_lexical_field(self):
        """Test that partial updates to new lexical fields are rejected.
        
        This test verifies that partial updates to new lexical fields are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'new_lexical_field': 'some string that signifies new lexical field'}], self.index)
        self.assertTrue(res.errors)
        self.assertIn("new_lexical_field of type str does not exist in the original document. We do not support adding new lexical fields in partial updates", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_invalid_field_name(self):
        """Test that partial updates to invalid field names are rejected.
        
        This test verifies that partial updates to invalid field names are rejected.
        """
        with pytest.raises(InvalidFieldNameError):
            res = self.config.document.partial_update_documents([{'_id': '2', 'marqo__': 1}], self.index)


    def test_partial_update_should_handle_mixed_numeric_map_updates(self):
        """Test updating maps with mix of additions and removals
        
        This test verifies that partial updates can correctly handle numeric maps
        with a mixture of operations:
        1. Updating existing key-value pairs
        2. Adding new key-value pairs
        
        The test performs updates on both integer maps and float maps, then
        verifies that all changes were applied correctly by retrieving the
        document and checking each individual key-value pair.
        """
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
        self.assertEqual(None, doc.get('float_map.d', None))
        self._assert_fields_unchanged(doc, ['int_map.a', 'int_map.b', 'int_map.c', 'float_map.c', 'float_map.e', 'float_map.d'])

    def test_partial_update_should_reject_invalid_map_values(self):
        """Test rejection of invalid value types in numeric maps
        
        This test verifies that partial updates reject invalid value types in numeric maps.
        """
        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_map': {
                'a': 'string',  # Invalid type
                'b': 2.5,      # Invalid type
                'c': True      # Invalid type
            }
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn("Unsupported field type <class 'str'> for field int_map in doc 2", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

        # Verify original values unchanged
        doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertEqual(1, doc['int_map.a'])
        self.assertEqual(2, doc['int_map.b'])
        self._assert_fields_unchanged(doc, ['int_map.a', 'int_map.b'])

    def test_partial_update_should_handle_multiple_docs(self):
        """Test updating multiple documents in one request
        
        This test verifies that partial updates can correctly handle multiple documents
        in a single request.
        """
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
        self._assert_fields_unchanged(doc2, ['int_field', 'float_map.c', 'float_map.d'])
        
        doc3 = tensor_search.get_document_by_id(self.config, self.index.name, '3')
        self.assertFalse(doc3['bool_field'])
        self.assertEqual(777, doc3['int_map.a'])
        self._assert_fields_unchanged(doc3, ['bool_field', 'int_map.a', 'int_map.b'])

    def test_partial_update_should_handle_duplicate_doc_ids(self):
        """Test handling of duplicate document IDs in update request
        
        This test verifies that partial updates can correctly handle duplicate document IDs
        in an update request.
        """
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
        self._assert_fields_unchanged(doc, ['int_field'])

    def test_partial_update_should_handle_non_existent_doc_id(self):
        """Test updating non-existent document
        
        This test verifies that partial updates can correctly handle non-existent document IDs.
        """
        res = self.config.document.partial_update_documents([{
            '_id': 'non_existent',
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)

    def test_partial_update_should_handle_none_id(self):
        """Test handling of None _id field
        
        This test verifies that partial updates can correctly handle None document IDs.
        """
        res = self.config.document.partial_update_documents([{
            '_id': None,
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('document _id must be a string type! received _id none of type `nonetype`', res.items[0].error.lower())
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_handle_missing_id(self):
        """Test handling of document without _id field
        
        This test verifies that partial updates can correctly handle documents
        without an _id field.
        """
        res = self.config.document.partial_update_documents([{
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn("'_id' is a required field", res.items[0].error.lower())
        self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_handle_empty_update_list(self):
        """Test handling of empty document list
        
        This test verifies that partial updates can correctly handle empty document lists.
        """
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
        self._assert_fields_unchanged(doc2, ['int_field'])

        doc3 = tensor_search.get_document_by_id(self.config, self.index.name, '3')
        self.assertTrue(doc3['bool_field'])
        self._assert_fields_unchanged(doc2, ['bool_field','int_field'])

        self.assertEqual(3, len(res.items))
        self.assertFalse(res.items[0].error)  # Valid doc
        self.assertFalse(res.items[1].error)  # Valid doc

        # Verify error responses for invalid docs
        self.assertIn("'_id' is a required field", res.items[2].error)  # Missing ID

    def test_partial_update_should_handle_nested_maps(self):
        """Test handling of nested maps in updates
        
        This test verifies that partial updates can correctly handle nested maps.
        """
        res = self.config.document.partial_update_documents([{
            '_id': '2',
            'int_map': {
                'nested': {
                    'too': 'deep'
                }
            }
        }], self.index)
        self.assertTrue(res.errors)
        self.assertEqual(400, res.items[0].status)
        self.assertIn('unsupported field type', res.items[0].error.lower())

    def test_partial_update_should_handle_empty_string_id(self):
        """Test handling of empty string as document ID
        
        This test verifies that partial updates can correctly handle empty string document IDs.
        """
        res = self.config.document.partial_update_documents([{
            '_id': '',
            'int_field': 100
        }], self.index)
        self.assertTrue(res.errors)
        self.assertIn("document id can't be empty", res.items[0].error.lower())

    def test_partial_update_should_handle_random_dict_field(self):
        """Test handling of random dictionary fields
        
        This test verifies that partial updates can correctly handle random dictionary fields.
        """
        res = self.config.document.partial_update_documents(
            [{
                '_id': '2',
                "random_field": {
                    "content1": "abcd",
                    "content2": "efgh"
                }
            }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('Unsupported field type', res.items[0].error)

    def test_partial_update_should_handle_random_field_type(self):
        """Test handling of random field types
        
        This test verifies that partial updates can correctly handle random field types.
        """
        res = self.config.document.partial_update_documents(
            [{
                '_id': '2',
                "random_field": None
            }], self.index)
        self.assertTrue(res.errors)
        self.assertIn('Unsupported field type', res.items[0].error)

    def test_partial_update_should_handle_empty_dict_field(self):
        """Test handling of empty dictionary fields
        
        This test verifies that partial updates can correctly handle empty dictionary fields.
        """
        res = self.config.document.partial_update_documents(
            [{
                '_id': '2',
                "float_map": {}
            }], self.index
        )
        self.assertFalse(res.errors)
        updated_doc = tensor_search.get_document_by_id(self.config, self.index.name, '2')
        self.assertIsNone(updated_doc.get('float_map.c', None))
        self.assertIsNone(updated_doc.get('float_map.d', None))
        self._assert_fields_unchanged(updated_doc, ['float_map.c', 'float_map.d'])

    def test_partial_update_should_reject_updating_dict_to_int_field(self):
        """Test that partial updates to dictionary fields are rejected when the value is an integer.
        
        This test verifies that partial updates to dictionary fields are rejected when the value is an integer.
        """
        res = self.config.document.partial_update_documents([
            {
                '_id': '2',
                "float_map": 100
            }
        ], self.index)
        self.assertTrue(res.errors)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)

    def test_updating_int_map_to_int(self):
        """Test that partial updates to int maps are successful.

        This test verifies that partial updates to int maps are rejected.
        """
        res = self.config.document.partial_update_documents([{'_id': '2', 'int_map': 100}], self.index)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertTrue(res.errors)
        self.assertEqual(400, res.items[0].status)
