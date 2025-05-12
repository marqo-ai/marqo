import copy
from typing import List, Dict, Any
from typing import Tuple

import pytest
from integ_tests.marqo_test import MarqoTestCase

from marqo.api.exceptions import InvalidFieldNameError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.semi_structured_vespa_index.marqo_field_types import MarqoFieldTypes
from marqo.tensor_search import tensor_search


class TestPartialUpdate(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        semi_structured_index_request = cls.unstructured_marqo_index_request(
            name='test_partial_update_semi_structured_14')
        cls.create_indexes([semi_structured_index_request])
        cls.index = cls.indexes[0]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.doc = {  # This document helps us understand / test behavior of partial updating fields when
            # Those fields don't exist in the original document
            '_id': '1',
            "string_array": ["aaa", "bbb"],
            "string_array2": ["123", "456"],
        }
        self.doc2 = {  # This document helps us test behavior of partial updating fields when
            # Those fields already exist in the original document
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
        self.doc3 = {  # This document doesn't contain any string arrays, it's used to test adding new string arrays
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
        self.field_to_field_type_doc1 = {
            'string_array': MarqoFieldTypes.STRING_ARRAY.value,
            'string_array2': MarqoFieldTypes.STRING_ARRAY.value,
        }
        self.field_to_field_type_doc2 = {
            'string_array': MarqoFieldTypes.STRING_ARRAY.value,
            'string_array2': MarqoFieldTypes.STRING_ARRAY.value,
            'int_map': MarqoFieldTypes.INT_MAP.value,
            'float_map': MarqoFieldTypes.FLOAT_MAP.value,
            'bool_field': MarqoFieldTypes.BOOL.value,
            'bool_field2': MarqoFieldTypes.BOOL.value,
            'custom_vector_field': MarqoFieldTypes.TENSOR.value,
            'multimodal_combo_field': MarqoFieldTypes.TENSOR.value,
            'tensor_field': MarqoFieldTypes.TENSOR.value,
            'tensor_subfield': MarqoFieldTypes.TENSOR.value,
            'short_string_field': MarqoFieldTypes.STRING.value,
            'long_string_field': MarqoFieldTypes.STRING.value,
            'lexical_field': MarqoFieldTypes.STRING.value,
            'int_map.a': MarqoFieldTypes.INT_MAP.value,
            'int_map.b': MarqoFieldTypes.INT_MAP.value,
            'float_map.c': MarqoFieldTypes.FLOAT_MAP.value,
            'float_map.d': MarqoFieldTypes.FLOAT_MAP.value,
            'float_field': MarqoFieldTypes.FLOAT.value,
            'int_field': MarqoFieldTypes.INT.value,
        }
        self.field_to_field_type_doc3 = {
            'bool_field': MarqoFieldTypes.BOOL.value,
            'bool_field2': MarqoFieldTypes.BOOL.value,
            'int_field': MarqoFieldTypes.INT.value,
            'float_field': MarqoFieldTypes.FLOAT.value,
            'int_map': MarqoFieldTypes.INT_MAP.value,
            'float_map': MarqoFieldTypes.FLOAT_MAP.value,
            'custom_vector_field': MarqoFieldTypes.TENSOR.value,
            'multimodal_combo_field': MarqoFieldTypes.TENSOR.value,
            'tensor_field': MarqoFieldTypes.TENSOR.value,
            'tensor_subfield': MarqoFieldTypes.TENSOR.value,
            'short_string_field': MarqoFieldTypes.STRING.value,
            'long_string_field': MarqoFieldTypes.STRING.value,
            'int_map.a': MarqoFieldTypes.INT_MAP.value,
            'int_map.b': MarqoFieldTypes.INT_MAP.value,
            'float_map.c': MarqoFieldTypes.FLOAT_MAP.value,
            'float_map.d': MarqoFieldTypes.FLOAT_MAP.value,
        }
        self.field_to_field_type_doc_minimal_doc = {
            "short_string_field": MarqoFieldTypes.STRING.value,
            "long_string_field": MarqoFieldTypes.STRING.value,
            "int_field": MarqoFieldTypes.INT.value,
            "float_field": MarqoFieldTypes.FLOAT.value,
            "bool_field": MarqoFieldTypes.BOOL.value,
            "bool_field2": MarqoFieldTypes.BOOL.value,
            "int_map": MarqoFieldTypes.INT_MAP.value,
            "float_map": MarqoFieldTypes.FLOAT_MAP.value,
            "int_map.key1": MarqoFieldTypes.INT_MAP.value,
            "int_map.key2": MarqoFieldTypes.INT_MAP.value,
            "int_map.key3": MarqoFieldTypes.INT_MAP.value,
            "float_map.key1": MarqoFieldTypes.FLOAT_MAP.value,
            "float_map.key2": MarqoFieldTypes.FLOAT_MAP.value,
            "float_map.key3": MarqoFieldTypes.FLOAT_MAP.value,
            "string_array": MarqoFieldTypes.STRING_ARRAY.value,
            "lexical_field": MarqoFieldTypes.STRING.value,
        }
        self.doc_to_field_type_map = {
            '1': self.field_to_field_type_doc1,
            '2': self.field_to_field_type_doc2,
            '3': self.field_to_field_type_doc3,
            'minimal_doc': self.field_to_field_type_doc_minimal_doc
        }

        self.add_documents(self.config, add_docs_params=AddDocsParams(
            index_name=self.index.name,
            docs=[self.doc, self.doc2, self.doc3],
            tensor_fields=['tensor_field', 'custom_vector_field', 'multimodal_combo_field'],
            mappings={
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

    def _assert_field_types(self, id: str, field_type_pairs: List[Tuple[str, MarqoFieldTypes]]):
        """
        Verify that the field types of a document match the expected types.

        This method retrieves the document from Vespa and checks that the field types
        match the expected types provided in the `field_type_pairs` list.

        Args:
            id (str): The ID of the document to check.
            field_type_pairs (List[Tuple[str, MarqoFieldTypes]]): A list of tuples where each tuple contains
                a field name and its expected type.

        Raises:
            AssertionError: If any field type does not match the expected type.
        """

        raw_vespa_doc = self.config.vespa_client.get_document(id, self.index.schema_name)
        vespa_fields = raw_vespa_doc.document.dict().get('fields')

        for field_name, field_type in field_type_pairs:
            expected_type = field_type.value if field_type is not None else None
            self.assertEqual(vespa_fields.get('marqo__field_types').get(field_name), expected_type,
                             f"Expected {field_name} to have type {expected_type} for document {id}")

        self._assert_field_types_not_changed(id, vespa_fields, field_type_pairs)

    def _assert_field_types_not_changed(self, id: str, vespa_fields: dict,
                                        excluded_fields: List[Tuple[str, MarqoFieldTypes]]):
        """
        Verify that field types remain unchanged except for the specified excluded fields.

        This helper method checks that all field types in the document match their expected values,
        excluding the fields that were intentionally modified during the test.

        Args:
            id (str): The document ID to check.
            vespa_fields (dict): The fields from the Vespa document.
            excluded_fields (List[Tuple[str, MarqoFieldTypes]]): List of field names that were intentionally modified and should be excluded from verification.
        """
        # Get the actual field types from the Vespa document
        actual_field_types = vespa_fields.get('marqo__field_types')
        # Get the expected field types from our mapping
        doc_to_field_type = self.doc_to_field_type_map[id]

        # Create a copy of the expected field types and remove excluded fields
        expected_field_types = copy.deepcopy(doc_to_field_type)

        # Process each excluded field
        for field_name, field_value in excluded_fields:
            # If field value is None, remove the field from expected types if it exists. We do this because 
            # fields of type None signify that the field was supposed to be removed from the document
            if field_value is None:
                if field_name in expected_field_types:
                    del expected_field_types[field_name]
            # Otherwise update the expected type with the new value. We do this because 
            # this field signifies a new field that has been added to the document. 
            else:
                if field_name not in expected_field_types:
                    expected_field_types[field_name] = field_value.value

        # Compare actual and expected field types
        is_expected_and_actual_equal = actual_field_types == expected_field_types
        # If they don't match, find and report the differences
        if not is_expected_and_actual_equal:
            # Find common fields between actual and expected
            intersection = set(actual_field_types.items()) & set(expected_field_types.items())
            # Find fields that are in expected but not in actual
            fields_present_in_expected_but_not_in_actual = set(expected_field_types.items()) - intersection
            # Find fields that are in actual but not in expected  
            fields_present_in_actual_but_not_in_expected = set(actual_field_types.items()) - intersection
            # Report any missing fields that should be present
            for field, _ in fields_present_in_expected_but_not_in_actual:
                self.fail(f"Field {field} is present in expected field types but not in actual field types")
            # Report any extra fields that shouldn't be present
            for field, _ in fields_present_in_actual_but_not_in_expected:
                self.fail(f"Field {field} is present in actual field types but not in expected field types")

    # Test update single field
    def test_partial_update_should_update_bool_field(self):
        """Test that boolean fields can be updated correctly via partial updates.
        
        This test verifies that boolean fields can be updated for multiple documents
        while ensuring other fields remain unchanged.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

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

                # Verify field type
                self._assert_field_types(id, [
                    ('bool_field', MarqoFieldTypes.BOOL)
                ])

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

                # Verify field type
                self._assert_field_types(id, [
                    ('int_field', MarqoFieldTypes.INT)
                ])

    def test_partial_update_to_non_existent_field(self):
        """Test that partial updates to non-existent fields are successful.
        
        This test case basically verifies that we can add new fields via partial updates
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Adding new field to document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents(
                    [{'_id': id, 'update_field_that_doesnt_exist': 500}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(500, doc['update_field_that_doesnt_exist'],
                                 f"Expected new field value to be 500 for document {id}")
                self._assert_fields_unchanged(doc, ['update_field_that_doesnt_exist'])

                # Verify field type
                self._assert_field_types(id, [
                    ('update_field_that_doesnt_exist', MarqoFieldTypes.INT)
                ])

    def test_partial_update_should_not_update_int_field_to_float(self):
        """Test that partial updates to int fields are rejected when the value is a float.
        
        This test verifies that partial updates to int fields are rejected when the value is a float.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update int field to float for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_field': 1.0}], self.index)
                self.assertTrue(res.errors)
                self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
                self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
                self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_update_float_field_to_float(self):
        """Test that partial updates to float fields are successful.
        
        This test verifies that partial updates to float fields are successful.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating float field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'float_field': 500.0}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(500.0, doc['float_field'], f"Expected float_field to be 500.0 for document {id}")
                self._assert_fields_unchanged(doc, ['float_field'])

                # Verify field type
                self._assert_field_types(id, [
                    ('float_field', MarqoFieldTypes.FLOAT)
                ])

    def test_partial_update_should_update_int_map(self):
        """Test that partial updates to int maps are successful.
        
        This test verifies that partial updates to int maps are successful.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating int map for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_map': {'c': 2, 'd': 3}}],
                                                                    self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(doc['int_map.c'], 2, f"Expected int_map.c to be 2 for document {id}")
                self.assertEqual(doc['int_map.d'], 3, f"Expected int_map.d to be 3 for document {id}")
                self.assertEqual(doc.get('int_map.a'), None, f"Expected int_map.a to be None for document {id}")
                self.assertEqual(doc.get('int_map.b'), None, f"Expected int_map.b to be None for document {id}")
                self._assert_fields_unchanged(doc, ['int_map'])

                # Verify field type
                self._assert_field_types(id, [
                    ('int_map', MarqoFieldTypes.INT_MAP),
                    ('int_map.c', MarqoFieldTypes.INT_MAP),
                    ('int_map.d', MarqoFieldTypes.INT_MAP),
                    ('int_map.a', None),
                    ('int_map.b', None)
                ])

    def test_partial_update_should_replace_int_map(self):
        """Test that partial updates to int maps where we change the keys inside
        a specific int map are successful
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Replacing int map keys for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_map': {'f': 2, 'g': 3}}],
                                                                    self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(doc['int_map.f'], 2, f"Expected int_map.f to be 2 for document {id}")
                self.assertEqual(doc['int_map.g'], 3, f"Expected int_map.g to be 3 for document {id}")
                self.assertEqual(doc.get('int_map.a'), None, f"Expected int_map.a to be None for document {id}")
                self.assertEqual(doc.get('int_map.b'), None, f"Expected int_map.b to be None for document {id}")
                self._assert_fields_unchanged(doc, ['int_map'])

                # Verify field type
                self._assert_field_types(id, [
                    ('int_map', MarqoFieldTypes.INT_MAP),
                    ('int_map.f', MarqoFieldTypes.INT_MAP),
                    ('int_map.g', MarqoFieldTypes.INT_MAP),
                    ('int_map.a', None),
                    ('int_map.b', None)
                ])

    def test_partial_update_should_update_int_map_with_new_value(self):
        """Test that partial updates to int maps with new values are successful."""
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Adding new key to int map for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_map': {
                    'd': 2
                }}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertIsNone(doc.get('int_map.a'), f"Expected int_map.a to be None for document {id}")
                self.assertIsNone(doc.get('int_map.b'), f"Expected int_map.b to be None for document {id}")
                self.assertEqual(doc['int_map.d'], 2, f"Expected int_map.d to be 2 for document {id}")

                # Verify field type
                self._assert_field_types(id, [
                    ('int_map', MarqoFieldTypes.INT_MAP),
                    ('int_map.d', MarqoFieldTypes.INT_MAP),
                    ('int_map.a', None),
                    ('int_map.b', None)
                ])

    def test_partial_update_should_update_float_map(self):
        """Test that partial updates to float maps are successful.
        
        This test verifies that partial updates to float maps are successful.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating float map for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'float_map': {'c': 2.0, 'd': 3.0}}],
                                                                    self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(doc['float_map.c'], 2.0, f"Expected float_map.c to be 2.0 for document {id}")
                self.assertEqual(doc['float_map.d'], 3.0, f"Expected float_map.d to be 3.0 for document {id}")
                self._assert_fields_unchanged(doc, ['float_map'])

                # Verify field type
                self._assert_field_types(id, [
                    ('float_map', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map.c', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map.d', MarqoFieldTypes.FLOAT_MAP)
                ])

    def test_partial_update_should_allow_changing_multiple_maps_in_same_request(self):
        """Test that partial updates to multiple maps in the same request are successful.
        
        This test verifies that partial updates to multiple maps in the same request are successful.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating multiple fields for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_field': 2, 'int_map': {
                    'a': 2,  # update int to int
                }, 'float_map': {
                    'c': 3.0,  # update float to float
                }, 'bool_field': False, 'float_field': 500.0}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(2, doc['int_field'], f"Expected int_field to be 2 for document {id}")
                self.assertFalse(doc['bool_field'], f"Expected bool_field to be False for document {id}")
                self.assertEqual(500.0, doc['float_field'], f"Expected float_field to be 500.0 for document {id}")
                self.assertEqual(doc['int_map.a'], 2, f"Expected int_map.a to be 2 for document {id}")
                self.assertIsNone(doc.get('int_map.b'), f"Expected int_map.b to be None for document {id}")
                self.assertEqual(doc['float_map.c'], 3.0, f"Expected float_map.c to be 3.0 for document {id}")
                self.assertIsNone(doc.get('float_map.d'), f"Expected float_map.d to be None for document {id}")
                self._assert_fields_unchanged(doc, ['int_map.a', 'int_map.b', 'float_map.d', 'float_map.c', 'int_field',
                                                    'bool_field', 'float_field'])

                # Verify field types
                self._assert_field_types(id, [
                    ('int_map', MarqoFieldTypes.INT_MAP),
                    ('int_map.a', MarqoFieldTypes.INT_MAP),
                    ('int_map.b', None),
                    ('float_map', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map.c', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map.d', None),
                    ('int_field', MarqoFieldTypes.INT),
                    ('bool_field', MarqoFieldTypes.BOOL),
                    ('float_field', MarqoFieldTypes.FLOAT)
                ])

    def test_partial_update_should_update_string_array(self):
        """Test that partial updates to string arrays are successful.
        
        This test verifies that partial updates to string arrays are successful.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating string array for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'string_array': ["ccc"]}], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(["ccc"], doc['string_array'], f"Expected string_array to be ['ccc'] for document {id}")
                self._assert_fields_unchanged(doc, ['string_array'])

                # Verify field type
                self._assert_field_types(id, [
                    ('string_array', MarqoFieldTypes.STRING_ARRAY)
                ])

    def test_partial_update_should_reject_new_string_array_field(self):
        """Test that partial updates to new string arrays are rejected."""
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Adding new string array for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'string_array3': ["ccc"]}], self.index)
                self.assertTrue(res.errors, f"Expected errors when adding new string array to document {id}")
                self.assertEqual(400, res.items[0].status)
                self.assertIn('Unstructured index updates only support updating existing string array fields',
                              res.items[0].error)

    def test_partial_update_should_allow_adding_new_string_string_array_field_if_present_in_other_docs_in_same_index(
            self):
        """Tests that partial updates allow adding new string / string array fields if they are present in some other document in the same index.

        For example, doc2 contains lexical_field and string_array. Hence when we try to add lexical_field and string_array to doc1, it should be allowed.
        """
        test_docs = [self.doc, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Adding field from another document to document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents(
                    [{'_id': id, "lexical_field": "some value 2", 'string_array': ["ccc"]}],
                    self.config.index_management.get_index(self.index.name))
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual("some value 2", doc['lexical_field'],
                                 f"Expected lexical_field to be 'some value 2' for document {id}")
                self.assertEqual(["ccc"], doc['string_array'], f"Expected string_array to be ['ccc'] for document {id}")
                self._assert_fields_unchanged(doc, ['lexical_field', 'string_array'])

                # Verify field types
                self._assert_field_types(id, [
                    ('lexical_field', MarqoFieldTypes.STRING),
                    ('string_array', MarqoFieldTypes.STRING_ARRAY)
                ])

    def test_partial_update_should_update_short_string(self):
        """Test that partial updates to short strings are successful."""
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating short string for document with ID {doc['_id']}"):
                id = doc['_id']
                index = self.config.index_management.get_index(self.index.name)
                res = self.config.document.partial_update_documents(
                    [{'_id': id, 'short_string_field': 'updated_short_string'}], index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual('updated_short_string', doc['short_string_field'],
                                 f"Expected short_string_field to be 'updated_short_string' for document {id}")
                self._assert_fields_unchanged(doc, ['short_string_field'])

                # Verify field type
                self._assert_field_types(id, [
                    ('short_string_field', MarqoFieldTypes.STRING)
                ])

    def test_partial_update_should_update_long_string(self):
        """Test that partial updates to long strings are successful."""
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating long string for document with ID {doc['_id']}"):
                id = doc['_id']
                index = self.config.index_management.get_index(self.index.name)
                res = self.config.document.partial_update_documents(
                    [{'_id': id, 'long_string_field': 'updated_long_string' * 10}], index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual('updated_long_string' * 10, doc['long_string_field'],
                                 f"Expected long_string_field to be updated for document {id}")
                self._assert_fields_unchanged(doc, ['long_string_field'])

                # Verify field type
                self._assert_field_types(id, [
                    ('long_string_field', MarqoFieldTypes.STRING)
                ])

    def test_partial_update_should_update_long_string_to_short_string(self):
        """Test that partial updates to long strings to short strings are successful."""
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating long string to short string for document with ID {doc['_id']}"):
                id = doc['_id']
                index = self.config.index_management.get_index(self.index.name)
                res = self.config.document.partial_update_documents(
                    [{'_id': id, 'long_string_field': 'short'}], index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual('short', doc['long_string_field'],
                                 f"Expected long_string_field to be 'short' for document {id}")
                self._assert_fields_unchanged(doc, ['long_string_field'])

                # Verify field type
                self._assert_field_types(id, [
                    ('long_string_field', MarqoFieldTypes.STRING)
                ])

    def test_partial_update_should_update_short_string_to_long_string(self):
        """Test that partial updates to short strings to long strings are successful."""

        id = self.doc2['_id']
        original_value = self.doc2["short_string_field"]
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{original_value}')
        self.assertEqual(2, len(res['hits']))

        index = self.config.index_management.get_index(self.index.name)
        res = self.config.document.partial_update_documents([{'_id': id, 'short_string_field': 'verylongstring' * 10}],
                                                            index)
        self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

        doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
        self.assertEqual('verylongstring' * 10, doc['short_string_field'],
                         f"Expected short_string_field to be updated for document {id}")
        self._assert_fields_unchanged(doc, ['short_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{original_value}')
        self.assertEqual(1, len(res['hits']))

    def test_partial_update_should_update_score_modifiers_and_version_uuid(self):
        """Test that partial updates to score modifiers are successful.
            Along with updating score modifiers, we also check that version_uuid changes since we are processing an update request that contains maps.
        """
        test_docs = [self.doc2, self.doc3]

        version_uuid = {}

        for doc in test_docs:
            with self.subTest(f"Updating score modifiers for document with ID {doc['_id']}"):
                id = doc['_id']
                # Doing a get to set the version_uuid in the version_uuid hashmap, which we'll check later to make sure it has changed after
                # processing an update request that contains maps
                raw_vespa_doc = self.config.vespa_client.get_document(id, self.index.schema_name)
                doc = raw_vespa_doc.document.dict().get('fields')
                self.assertIsNotNone(doc.get('marqo__version_uuid'))  # version_uuid should be present.
                version_uuid[id] = doc.get('marqo__version_uuid')

                res = self.config.document.partial_update_documents([{'_id': id, 'int_map': {
                    'a': 2,  # update int to int
                    'd': 5,  # new entry in int map
                }, 'float_map': {
                    'c': 3.0,  # update float to float
                }, 'new_int': 1,  # new int field
                                                                      'new_float': 2.0,  # new float field
                                                                      'new_map': {'a': 1, 'b': 2.0},  # new map field
                                                                      }], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                # Verify field types
                field_type_pairs = [
                    ('int_map', MarqoFieldTypes.INT_MAP),
                    ('int_map.a', MarqoFieldTypes.INT_MAP),
                    ('int_map.d', MarqoFieldTypes.INT_MAP),
                    ('float_map', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map.c', MarqoFieldTypes.FLOAT_MAP),
                    ('new_int', MarqoFieldTypes.INT),
                    ('new_float', MarqoFieldTypes.FLOAT),
                    ('new_map.a', MarqoFieldTypes.INT_MAP),
                    ('new_map.b', MarqoFieldTypes.FLOAT_MAP),
                    ('new_map', MarqoFieldTypes.FLOAT_MAP),
                    ('int_map.b', None),
                    ('float_map.d', None)
                ]

                self._assert_field_types(id, field_type_pairs)

                # Also check score modifiers values
                raw_vespa_doc = self.config.vespa_client.get_document(id, self.config.index_management.get_index(
                    self.index.name).schema_name)
                doc = raw_vespa_doc.document.dict().get('fields')
                int_field_val = self.id_to_doc[id].get('int_field', 0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_field'], float(int_field_val))
                float_field_val = self.id_to_doc[id].get('float_field', 0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_field'], float(float_field_val))
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.a'], 2.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], 3.0)

                # assert that the map keys that must've been removed don't exist in marqo__score_modifiers
                self.assertEqual(doc['marqo__score_modifiers']['cells'].get('int_map.b', None), None)
                self.assertEqual(doc['marqo__score_modifiers']['cells'].get('float_map.d', None), None)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_int'], 1.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_float'], 2.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_map.a'], 1.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_map.b'], 2.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.d'], 5.0)
                # Assert that after processing an update request that contains map fields, version uuid changes
                self.assertNotEqual(doc.get('marqo__version_uuid'), version_uuid.get(id))  # version_uuid should change

    def test_partial_update_should_add_score_modifiers(self):
        """
        Test that partial updates which specifically add new fields reflect properly in score modifiers tensors.
        Along with updating score modifiers, we also check that version_uuid changes since we are processing an update request that contains maps.
        """
        test_docs = [self.doc, self.doc2, self.doc3]
        version_uuid = {}

        for doc in test_docs:
            with self.subTest(f"Adding score modifiers for document with ID {doc['_id']}"):
                id = doc['_id']
                # Doing a get to set the version_uuid in the version_uuid hashmap, which we'll check later to make sure it has changed after
                # processing an update request that contains maps
                raw_vespa_doc = self.config.vespa_client.get_document(id, self.index.schema_name)
                doc = raw_vespa_doc.document.dict().get('fields')
                self.assertIsNotNone(doc.get('marqo__version_uuid'))  # version_uuid should be present.
                version_uuid[id] = doc.get('marqo__version_uuid')

                # Create a document with existing fields first to verify we're only adding
                original_doc = tensor_search.get_document_by_id(self.config, self.index.name, id)

                # Perform update with only additions, not replacements
                res = self.config.document.partial_update_documents([{
                    '_id': id,
                    'int_map_2': {
                        'd': 5,  # adding entirely new map
                        'e': 6,
                    },
                    'float_map_2': {
                        'f': 4.0,  # adding entirely new map
                    },
                    'new_int': 1,  # new int field
                    'new_float': 2.0,  # new float field
                }], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                # Verify field types
                # Verify field types using helper method
                field_type_pairs = [
                    ('int_map_2.d', MarqoFieldTypes.INT_MAP),
                    ('int_map_2', MarqoFieldTypes.INT_MAP),
                    ('int_map_2.e', MarqoFieldTypes.INT_MAP),
                    ('float_map_2.f', MarqoFieldTypes.FLOAT_MAP),
                    ('float_map_2', MarqoFieldTypes.FLOAT_MAP),
                    ('new_int', MarqoFieldTypes.INT),
                    ('new_float', MarqoFieldTypes.FLOAT)
                ]
                self._assert_field_types(id, field_type_pairs)

                res = self.config.vespa_client.get_document(id, self.config.index_management.get_index(
                    self.index.name).schema_name)
                doc = res.document.dict().get('fields')

                # Check score modifiers values for new fields
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map_2.d'], 5.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map_2.e'], 6.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map_2.f'], 4.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_int'], 1.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['new_float'], 2.0)

                # Verify that the version_uuid has changed. Only applicable for cases where we process update requests
                # that contain maps in them.
                self.assertNotEqual(doc.get('marqo__version_uuid'), version_uuid.get(id))  # version_uuid should change

                # Verify original fields are preserved (with conditional checks)
                int_field_val = self.id_to_doc[id].get('int_field')
                if int_field_val:
                    self.assertEqual(doc['marqo__score_modifiers']['cells']['int_field'], float(int_field_val))
                float_field_val = self.id_to_doc[id].get('float_field', 0)
                if float_field_val:
                    self.assertEqual(doc['marqo__score_modifiers']['cells']['float_field'], float(float_field_val))
                int_map_a_val = self.id_to_doc[id].get('int_map', {}).get('a', 0)
                if int_map_a_val:
                    self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.a'], float(int_map_a_val))
                float_map_c_val = self.id_to_doc[id].get('float_map', {}).get('c', 0)
                if float_map_c_val:
                    self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], float(float_map_c_val))

    def test_partial_update_only_update_existing_score_modifiers(self):
        """
         Test that partial updates which specifically change the existing keys inside existing maps
         reflect properly in score modifiers tensors.
         Along with updating score modifiers, we also check that version_uuid changes since we are processing an update request that contains maps.
         """
        test_docs = [self.doc2, self.doc3]
        version_uuid = {}

        for doc in test_docs:
            with self.subTest(f"Updating existing score modifiers for document with ID {doc['_id']}"):
                id = doc['_id']
                # Doing a get to set the version_uuid in the version_uuid hashmap, which we'll check later to make sure it has changed after
                # processing an update request that contains maps
                raw_vespa_doc = self.config.vespa_client.get_document(id, self.index.schema_name)
                doc = raw_vespa_doc.document.dict().get('fields')
                self.assertIsNotNone(doc.get('marqo__version_uuid'))  # version_uuid should be present.
                version_uuid[id] = doc.get('marqo__version_uuid')

                # Create a document with existing fields first to verify we're only adding
                original_doc = tensor_search.get_document_by_id(self.config, self.index.name, id)

                # Perform update with only additions, not replacements
                res = self.config.document.partial_update_documents([{
                    '_id': id,
                    "int_map": {"a": 3, "b": 4},
                    "float_map": {"c": 3.0, "d": 4.0},
                }], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                res = self.config.vespa_client.get_document(id,
                                                            self.config.index_management.get_index(
                                                                self.index.name).schema_name)
                doc = res.document.dict().get('fields')
                # Verify original fields are preserved
                int_field_val = self.id_to_doc[id].get('int_field', 0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_field'], float(int_field_val))
                float_field_val = self.id_to_doc[id].get('float_field', 0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_field'], float(float_field_val))
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.a'], 3.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['int_map.b'], 4.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.c'], 3.0)
                self.assertEqual(doc['marqo__score_modifiers']['cells']['float_map.d'], 4.0)

                # Verify that the version_uuid has changed. Only applicable for cases where we process update requests
                # that contain maps in them.
                self.assertNotEqual(doc.get('marqo__version_uuid'), version_uuid.get(id))  # version_uuid should change

    def test_partial_update_should_add_new_fields(self):
        """Test that partial updates to new fields are successful."""
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Adding new fields to document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'new_field': 500, 'new_float': 500.0,
                                                                      'new_int_map': {'a': 2},
                                                                      'new_bool_field': True,
                                                                      'new_float_field': 10.0
                                                                      }], self.config.index_management.get_index(
                    self.index.name))
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self._assert_fields_unchanged(doc, [])
                self.assertEqual(500.0, doc['new_float'], f"Expected new_float to be 500.0 for document {id}")
                self.assertEqual(2, doc['new_int_map.a'], f"Expected new_int_map.a to be 2 for document {id}")
                self.assertEqual(500, doc['new_field'], f"Expected new_field to be 500 for document {id}")
                self.assertEqual(True, doc['new_bool_field'], f"Expected new_bool_field to be True for document {id}")
                self.assertEqual(10.0, doc['new_float_field'], f"Expected new_float_field to be 10.0 for document {id}")

                # Verify field types
                field_type_pairs = [
                    ('new_field', MarqoFieldTypes.INT),
                    ('new_float', MarqoFieldTypes.FLOAT),
                    ('new_int_map.a', MarqoFieldTypes.INT_MAP),
                    ('new_int_map', MarqoFieldTypes.INT_MAP),
                    ('new_bool_field', MarqoFieldTypes.BOOL),
                    ('new_float_field', MarqoFieldTypes.FLOAT)
                ]
                self._assert_field_types(id, field_type_pairs)

    # Reject any tensor field change
    def test_partial_update_should_reject_tensor_field(self):
        """Test that partial updates to tensor fields are rejected.
        
        This test verifies that partial updates to tensor fields are rejected.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update tensor field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'tensor_field': 'new_title'}],
                                                                    self.index)
                self.assertTrue(res.errors)
                self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
                self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
                self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_multi_modal_field_subfield(self):
        """Test that partial updates to tensor subfields are rejected.
        
        This test verifies that partial updates to tensor subfields are rejected.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update tensor subfield for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'tensor_subfield': 'new_description'}],
                                                                    self.index)
                self.assertTrue(res.errors)
                self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
                self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
                self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_custom_vector_field(self):
        """Test that partial updates to custom vector fields are rejected.
        
        This test verifies that partial updates to custom vector fields are rejected.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update custom vector field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'custom_vector_field': {
                    "content": "efgh",
                    "vector": [1.0] * 32
                }}], self.index)
                self.assertTrue(res.errors)
                self.assertEqual(400, res.items[0].status)
                self.assertIn(f"Unsupported field type <class 'str'> for field custom_vector_field in doc {id}. "
                              "We only support int and float types for map values when updating a document",
                              res.items[0].error)

    def test_partial_update_should_reject_multimodal_combo_field(self):
        """Test that partial updates to multimodal combo fields are rejected.
        
        This test verifies that partial updates to multimodal combo fields are rejected.
        """
        test_docs = [self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update multimodal combo field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'multimodal_combo_field': {
                    "tensor_field": "new_title",
                    "tensor_subfield": "new_description"
                }}], self.index)
                self.assertTrue(res.errors)
                self.assertIn(f"Unsupported field type <class 'str'> for field multimodal_combo_field in doc {id}",
                              res.items[0].error)
                self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_numeric_array_field_type(self):
        """Test that partial updates to numeric array fields are rejected.
        
        This test verifies that partial updates to numeric array fields are rejected.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update numeric array field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{'_id': id, 'int_array': [1, 2, 3]}], self.index)
                self.assertTrue(res.errors)
                self.assertIn("Unstructured index updates only support updating existing string array fields",
                              res.items[0].error)
                self.assertEqual(400, res.items[0].status)

    def test_partial_update_should_reject_new_lexical_field(self):
        """Test that partial updates to new lexical fields are rejected.
        
        This test verifies that partial updates to new lexical fields are rejected.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update new lexical field for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents(
                    [{'_id': id, 'new_lexical_field': 'some string that signifies new lexical field'}], self.index)
                self.assertTrue(res.errors)
                self.assertIn(
                    "new_lexical_field of type str does not exist in the original document. Marqo does not support adding new lexical fields in partial updates",
                    res.items[0].error)
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
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Updating mixed numeric maps for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{
                    '_id': id,
                    'int_map': {
                        'a': 10,  # Update existing
                        'c': 3,  # Add new
                        'b': 20  # Update existing
                    },
                    'float_map': {
                        'c': 10.5,  # Update existing
                        'e': 5.5  # Add new
                    }
                }], self.index)
                self.assertFalse(res.errors, f"Expected no errors when updating document {id}")

                doc = tensor_search.get_document_by_id(self.config, self.index.name, id)
                self.assertEqual(10, doc['int_map.a'])
                self.assertEqual(20, doc['int_map.b'])
                self.assertEqual(3, doc['int_map.c'])
                self.assertEqual(10.5, doc['float_map.c'])
                self.assertEqual(5.5, doc['float_map.e'])
                self.assertEqual(None, doc.get('float_map.d', None))
                self._assert_fields_unchanged(doc, ['int_map.a', 'int_map.b', 'int_map.c', 'float_map.c', 'float_map.e',
                                                    'float_map.d'])

                # Verify field types
                self._assert_field_types(
                    id,
                    [
                        ('int_map.a', MarqoFieldTypes.INT_MAP),
                        ('int_map', MarqoFieldTypes.INT_MAP),
                        ('int_map.b', MarqoFieldTypes.INT_MAP),
                        ('int_map.c', MarqoFieldTypes.INT_MAP),
                        ('float_map.c', MarqoFieldTypes.FLOAT_MAP),
                        ('float_map.e', MarqoFieldTypes.FLOAT_MAP),
                        ('float_map.d', None),
                        ('float_map', MarqoFieldTypes.FLOAT_MAP)
                    ]
                )

    def test_partial_update_should_reject_invalid_map_values(self):
        """Test rejection of invalid value types in numeric maps
        
        This test verifies that partial updates reject invalid value types in numeric maps.
        """
        test_docs = [self.doc, self.doc2, self.doc3]

        for doc in test_docs:
            with self.subTest(f"Attempting to update invalid map values for document with ID {doc['_id']}"):
                id = doc['_id']
                res = self.config.document.partial_update_documents([{
                    '_id': id,
                    'int_map': {
                        'a': 'string',  # Invalid type
                        'b': 2.5,  # Invalid type
                        'c': True  # Invalid type
                    }
                }], self.index)
                self.assertTrue(res.errors)
                self.assertIn(f"Unsupported field type <class 'str'> for field int_map in doc {id}", res.items[0].error)
                self.assertEqual(400, res.items[0].status)

                # Verify original values unchanged
                get_docs = tensor_search.get_document_by_id(self.config, self.index.name, id)
                if doc.get('int_map.a',
                           None):  # Only assert that the original values are unchanged if they exist in the original document
                    self.assertEqual(1, get_docs['int_map.a'])
                if get_docs.get('int_map.b',
                                None):  # Only assert that the original values are unchanged if they exist in the original document
                    self.assertEqual(2, get_docs['int_map.b'])
                self._assert_fields_unchanged(get_docs, ['int_map.a', 'int_map.b'])

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
        self._assert_fields_unchanged(doc2, ['int_field', 'float_map.c', 'float_map.d'])

        doc3 = tensor_search.get_document_by_id(self.config, self.index.name, '3')
        self.assertFalse(doc3['bool_field'])
        self.assertEqual(777, doc3['int_map.a'])
        self._assert_fields_unchanged(doc3, ['bool_field', 'int_map.a', 'int_map.b'])

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
        self.assertIn('document _id must be a string type! received _id none of type `nonetype`',
                      res.items[0].error.lower())
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
        self._assert_fields_unchanged(doc2, ['bool_field', 'int_field'])

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
        self._assert_field_types('2', [('float_map', None), ('float_map.c', None), ('float_map.d', None)])
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

    def test_updating_int_to_int_map(self):
        """Test that partial updates changing int field to int maps are rejected.
        """

        res = self.config.document.partial_update_documents([{'_id': '2', 'int_field': {'a': 100}}], self.index)
        print(res)
        raw_vespa_doc = self.config.vespa_client.get_document('2', self.index.schema_name)
        print(raw_vespa_doc)
        self.assertTrue(res.errors)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertEqual(400, res.items[0].status)

    def test_updating_non_existent_document_with_maps(self):
        """
        Test updating a non-existent document with maps.

        This test verifies that attempting to update a non-existent document with a map field
        results in an error response. It checks a special handling we have added for documents in update requests which contain maps fields.
        These documents are not originally present in Vespa and Marqo must return appropriate response for them.
        """
        res = self.config.document.partial_update_documents([{'_id': '4', 'metadata': {'key1': 2}}], self.index)
        self.assertTrue(res.errors)
        self.assertEqual(400, res.items[0].status)
        self.assertIn("Marqo vector store couldn't update the document. Please see", res.items[0].error)
        self.assertIn('reference/api/documents/update-documents/#response', res.items[0].error)

    def test_partial_update_adding_all_field_types_to_minimal_document(self):
        """Test adding all possible field types to a minimal document via partial update.
        
        This test:
        1. Creates a minimal document with just an ID
        2. Performs a partial update to add all supported field types
        3. Verifies that all field types are correctly set in the document
        """
        # Create a minimal document with just an ID
        minimal_doc = {
            "_id": "minimal_doc"
        }

        # Add the minimal document to the index
        self.add_documents(self.config, add_docs_params=AddDocsParams(
            index_name=self.index.name,
            docs=[minimal_doc],
            tensor_fields=[]
        ))

        # Verify the document exists
        doc_before_update = tensor_search.get_document_by_id(self.config, self.index.name, "minimal_doc")
        self.assertEqual("minimal_doc", doc_before_update["_id"])

        # Perform a partial update to add all supported field types
        update_fields = {
            "_id": "minimal_doc",
            "short_string_field": "short string value",
            "long_string_field": "This is a very long string value " * 10,
            "int_field": 42,
            "float_field": 3.14159,
            "bool_field": True,
            "bool_field2": False,
            "int_map": {"key1": 1, "key2": 2, "key3": 3},
            "float_map": {"key1": 1.1, "key2": 2.2, "key3": 3.3},
            "string_array": ["value1", "value2", "value3"],
            "lexical_field": "lexical field value"
        }

        res = self.config.document.partial_update_documents([update_fields], self.index)
        self.assertFalse(res.errors,
                         f"Expected no errors when updating document, got: {res.items[0].error if res.errors else ''}")

        # Retrieve the updated document
        updated_doc = tensor_search.get_document_by_id(self.config, self.index.name, "minimal_doc")

        # Verify all fields were added with correct values
        self.assertEqual("short string value", updated_doc["short_string_field"])
        self.assertEqual("This is a very long string value " * 10, updated_doc["long_string_field"])
        self.assertEqual(42, updated_doc["int_field"])
        self.assertEqual(3.14159, updated_doc["float_field"])
        self.assertTrue(updated_doc["bool_field"])
        self.assertFalse(updated_doc["bool_field2"])
        self.assertEqual(1, updated_doc["int_map.key1"])
        self.assertEqual(2, updated_doc["int_map.key2"])
        self.assertEqual(3, updated_doc["int_map.key3"])
        self.assertEqual(1.1, updated_doc["float_map.key1"])
        self.assertEqual(2.2, updated_doc["float_map.key2"])
        self.assertEqual(3.3, updated_doc["float_map.key3"])
        self.assertEqual(["value1", "value2", "value3"], updated_doc["string_array"])
        self.assertEqual("lexical field value", updated_doc["lexical_field"])

        # Verify field types
        self._assert_field_types("minimal_doc", [
            ("short_string_field", MarqoFieldTypes.STRING),
            ("long_string_field", MarqoFieldTypes.STRING),
            ("int_field", MarqoFieldTypes.INT),
            ("float_field", MarqoFieldTypes.FLOAT),
            ("bool_field", MarqoFieldTypes.BOOL),
            ("bool_field2", MarqoFieldTypes.BOOL),
            ("int_map", MarqoFieldTypes.INT_MAP),
            ("int_map.key1", MarqoFieldTypes.INT_MAP),
            ("int_map.key2", MarqoFieldTypes.INT_MAP),
            ("int_map.key3", MarqoFieldTypes.INT_MAP),
            ("float_map", MarqoFieldTypes.FLOAT_MAP),
            ("float_map.key1", MarqoFieldTypes.FLOAT_MAP),
            ("float_map.key2", MarqoFieldTypes.FLOAT_MAP),
            ("float_map.key3", MarqoFieldTypes.FLOAT_MAP),
            ("string_array", MarqoFieldTypes.STRING_ARRAY),
            ("lexical_field", MarqoFieldTypes.STRING)
        ])
