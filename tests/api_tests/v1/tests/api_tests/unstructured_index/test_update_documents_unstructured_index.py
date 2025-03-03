import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase


class TestUpdateDocumentsInUnstructuredIndex(MarqoTestCase):
    """
    Support for partial updates for unstructured indexes was added in 2.16.0. Unstructured indexes are internally implemented as semi-structured indexes.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "random/small",
                "normalizeEmbeddings": False,
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name]
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

        add_docs_response = cls.client.index(cls.text_index_name).add_documents(documents = text_docs, mappings = mappings, tensor_fields = tensor_fields)

        cls.assertFalse(add_docs_response["errors"])

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_update_document_with_ids(self):

        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '1',
                'bool_field': False,
                'update_field_that_doesnt_exist': 500,
                'int_field': 1,
                'float_field': 500.0,
                'int_map': {
                    'a': 2,
                },
                'float_map': {
                    'c': 3.0,
                },
                'string_array': ["ccc"]
            }]
        )

        assert update_docs_response["errors"] == False

        get_docs_response = self.client.index(self.text_index_name).get_document(document_id = '1')

        self.assertEqual(get_docs_response['bool_field'], False)
        self.assertEqual(get_docs_response['int_field'], 1)
        self.assertEqual(get_docs_response['float_field'], 500.0)
        self.assertEqual(get_docs_response['int_map.a'], 2)
        self.assertEqual(get_docs_response['float_map.c'], 3.0)
        self.assertEqual(get_docs_response['string_array'], ["ccc"])
        self.assertEqual(get_docs_response['update_field_that_doesnt_exist'], 500)
        self.assertEqual(get_docs_response['string_array2'], ["123", "456"])

    def test_update_document_with_ids_change_field_type(self):

        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '1',
                'bool_field': False,
                'update_field_that_doesnt_exist': 500,
                'int_field': 1,
                'float_field': 500, # The request is same as the test case test_update_document_with_ids, except the float_field value is changed to int. This will result in a 412 condition check failed error.
                'int_map': {
                    'a': 2,
                },
                'float_map': {
                    'c': 3.0,
                },
                'string_array': ["ccc"]
            }]
        )

        self.assertTrue(update_docs_response["errors"])

        self.assertEqual(update_docs_response['items'][0]['status'], 400)
        self.assertIn("Marqo vector store couldn't update the document. Please see", update_docs_response['items'][0]['message'])
        self.assertIn("reference/api/documents/update-documents/#response", update_docs_response['items'][0]['message'])

    def test_update_document_with_changes_in_score_modifiers(self):
        """Test that score modifiers are correctly updated during partial document updates.
        
        This test verifies that:
        1. Score modifiers are properly updated when numeric fields are modified
        2. New numeric fields are correctly added to score modifiers
        3. The updated score modifiers affect search results as expected
        """
        # First add a document to update
        """Test updating a document with new fields and updating existing fields."""
        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '1',
                'int_map': {
                    'a': 2,  # update int to int
                    'd': 5,  # new entry in int map
                },
                'float_map': {
                    'c': 3.0,  # update float to float
                },
                'new_int': 1,  # new int field
                'new_float': 2.0,  # new float field
                'new_map': {'a': 1, 'b': 2.0},  # new map field
            }]
        )

        self.assertFalse(update_docs_response["errors"])

        # Get the document to verify updates
        updated_doc = self.client.index(self.text_index_name).get_document(document_id='1')
        self.assertEqual(updated_doc['int_map.a'], 2)
        self.assertEqual(updated_doc['int_map.d'], 5)
        self.assertEqual(updated_doc['float_map.c'], 3.0)
        self.assertEqual(updated_doc['new_int'], 1)
        self.assertEqual(updated_doc['new_float'], 2.0)
        self.assertEqual(updated_doc['new_map.a'], 1)
        self.assertEqual(updated_doc['new_map.b'], 2.0)

        # Test that score modifiers work correctly with the updated fields
        # First search without score modifier to get base score
        base_search_result = self.client.index(self.text_index_name).search("title")
        self.assertTrue(len(base_search_result["hits"]) > 0, "No search results found")
        base_score = base_search_result["hits"][0]["_score"]
        
        # Search with score modifier weight=0 (should not change score)
        search_result_weight_0 = self.client.index(self.text_index_name).search("title", score_modifiers={
            "add_to_score": [{"field_name": "int_map.d", "weight": 0}]
        })
        self.assertAlmostEqual(search_result_weight_0["hits"][0]["_score"], base_score, places=5)
        
        # Search with score modifier weight=1 (should add int_map.d value to score)
        search_result_weight_1 = self.client.index(self.text_index_name).search("title", score_modifiers={
            "add_to_score": [{"field_name": "int_map.d", "weight": 1}]
        })
        # The score should be increased by weight * field_value = 1 * 5 = 5
        self.assertAlmostEqual(
            search_result_weight_1["hits"][0]["_score"], 
            base_score + 5, 
            places=5
        )
        
        # Verify the field value is actually 5
        self.assertEqual(search_result_weight_1["hits"][0]["int_map.d"], 5)
        
        # Now update the document again to change the score modifier field
        update_docs_response_2 = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '1',
                'int_map': {
                    'd': 10,  # update the value from 5 to 10
                }
            }]
        )
        
        self.assertFalse(update_docs_response_2["errors"])
        
        # Search again with score modifier weight=1 after update
        search_result_after_update = self.client.index(self.text_index_name).search("title", score_modifiers={
            "add_to_score": [{"field_name": "int_map.d", "weight": 1}]
        })
        
        # Verify the field value is now 10
        self.assertEqual(search_result_after_update["hits"][0]["int_map.d"], 10)
        
        # The score should now be increased by weight * new_field_value = 1 * 10 = 10
        self.assertAlmostEqual(
            search_result_after_update["hits"][0]["_score"], 
            base_score + 10, 
            places=5
        )


