import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.16.0')
class TestUpdateDocumentsUnstructured2_16(BaseCompatibilityTestCase):
    """
    Partial updates for unstructured indexes was introduced in 2.16.0. This is a backwards compatibility test which runs on
    from_version >= 2.16.0. It will test that any future releases post 2.16.0 don't break the partial update functionality for unstructured indexes.
    """
    unstructured_index_name = "test_update_documents_unstructured_index"
    indexes_to_test_on = [
        {
            "indexName": unstructured_index_name,
            "type": "unstructured",
            "model": "random/small",
            "normalizeEmbeddings": True,
    }]

    text_docs = [{
                '_id': '1',
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
            }]


    mappings = {
        "custom_vector_field": {"type": "custom_vector"},
        "multimodal_combo_field": {
            "type": "multimodal_combination",
            "weights": {"tensor_field": 1.0, "tensor_subfield": 2.0}
        }
    }

    tensor_fields = ['tensor_field', 'custom_vector_field', 'multimodal_combo_field']

    partial_update_test_cases = [{
        '_id': '1',
        'bool_field': False,
        'update_field_that_doesnt_exist': 500,
        'int_field': 1,
        'float_field': 500.0,
        'int_map': {
            'a': 2,  # update int to int
        },
        'float_map': {
            'c': 3.0,  # update float to int
        },
        'string_array': ["ccc"]
        }]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().setUpClass()


    def prepare(self):
        self.logger.debug(f"Creating indexes {self.indexes_to_test_on} in test case: {self.__class__.__name__}")
        self.create_indexes(self.indexes_to_test_on)

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            with self.subTest(indexName=index_name):
                self.client.index(index_name = index['indexName']).add_documents(documents = self.text_docs, mappings = self.mappings, tensor_fields = self.tensor_fields)


        all_results = {}
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}
            for docs in self.text_docs:
                with self.subTest(indexName=index_name, doc_id = docs['_id']):
                    get_docs_result = self.client.index(index_name).get_document(document_id = docs['_id'])
                    all_results[index_name] = get_docs_result

    def test_update_doc(self):
        self.logger.info(f"Running test_update_doc on {self.__class__.__name__}")

        test_failures = []

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                with self.subTest(indexName = index_name):
                    result = self.client.index(index_name).update_documents(
                        self.partial_update_test_cases
                    )
                self.logger.debug(f"Printing result {result}")

            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

            assert result["index_name"] == self.unstructured_index_name
            assert result["errors"] == False

            for test_cases in self.partial_update_test_cases:
                doc_id = test_cases['_id']
                get_docs_result = self.client.index(index_name).get_document(document_id = doc_id)
                self._assert_updates_have_happened(get_docs_result, test_cases)

        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")
    def _assert_updates_have_happened(self, result, partial_update_test_case):
        """
        Verifies that document updates have been correctly applied.
        
        Args:
            result (dict): The document retrieved after update operation
            partial_update_test_case (dict): The update operation that was applied
            
        Raises:
            AssertionError: If any field doesn't match the expected updated value
        """
        self.logger.debug(f"Verifying updates for document {partial_update_test_case.get('_id')}")
        
        for field, expected_value in partial_update_test_case.items():
            if field == "_id":
                continue
                
            # Handle dictionary fields
            if isinstance(expected_value, dict):
                self.logger.debug(f"Checking dictionary field: {field}")
                # For dictionary fields, we need to check each key separately
                for key, value in expected_value.items():
                    nested_field = f"{field}.{key}"
                    if result.get(nested_field) != value:
                        self.fail(f"Nested dictionary field {nested_field} does not match expected value. "
                                  f"Got: {result.get(nested_field)}, Expected: {value}")
            # Handle regular fields
            else:
                self.logger.debug(f"Checking regular field: {field}")
                if result.get(field) != expected_value:
                    self.fail(f"Field {field} does not match expected value. "
                              f"Got: {result.get(field)}, Expected: {expected_value}")
                    
        self.logger.debug(f"All updates verified successfully for document {partial_update_test_case.get('_id')}")
