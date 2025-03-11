import uuid
import threading
import time
from datetime import datetime

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

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_update_document_with_ids(self):

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

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents = text_docs, mappings = mappings, tensor_fields = tensor_fields)

        self.assertFalse(add_docs_response["errors"])

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

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents = text_docs, mappings = mappings, tensor_fields = tensor_fields)

        self.assertFalse(add_docs_response["errors"])

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

    def test_concurrent_partial_update_requests(self):
        """Test concurrent updates to different fields of the same document.
        
        This test verifies that:
        1. Multiple threads can update different fields of the same document concurrently
        2. Updates are properly applied without conflicts
        3. The final document state reflects one of the updates correctly
        """
        # First add a document to update
        text_docs = [{
            '_id': '3',
            'tensor_field': 'concurrent update test',
            'description': 'This document will be updated by multiple threads',
            'int_field': 100,
            'float_field': 100.0,
        }]

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents=text_docs, mappings={}, tensor_fields=['tensor_field', 'description'])
        self.assertFalse(add_docs_response["errors"])

        def update_rank_thread(index_name, rank_values):
            for i, new_rank in enumerate(rank_values):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Rank update {i+1}/{len(rank_values)}: Setting rank to {new_rank}")
                r = self.client.index(index_name).update_documents([{'_id': '3', 'score_map': {'rank': new_rank}}])
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Rank update {i+1} complete. Response: {r}")
                time.sleep(0.5)  # Small delay between updates

        def update_popularity_thread(index_name, popularity_values):
            for i, new_pop in enumerate(popularity_values):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Popularity update {i+1}/{len(popularity_values)}: Setting popularity to {new_pop}")
                r = self.client.index(index_name).update_documents([{'_id': '3', 'score_map': {'popularity': new_pop}}])
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Popularity update {i+1} complete. Response: {r}")
                time.sleep(0.5)  # Same delay now for both threads

        rank_values = [0.85, 0.87, 0.90, 0.82, 0.88]
        popularity_values = [0.72, 0.75, 0.79, 0.81, 0.78]

        rank_thread = threading.Thread(target=update_rank_thread, args=(self.text_index_name, rank_values))
        pop_thread = threading.Thread(target=update_popularity_thread, args=(self.text_index_name, popularity_values))

        rank_thread.start()
        pop_thread.start()

        rank_thread.join()
        pop_thread.join()

        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Both updates completed")

        # Get the document to verify updates
        updated_doc = self.client.index(self.text_index_name).get_document(document_id='3')
        
        # Check that only one of the fields is present (due to concurrent updates)
        has_rank = 'score_map.rank' in updated_doc
        has_popularity = 'score_map.popularity' in updated_doc
        
        # Either rank or popularity should be present, but not both
        self.assertTrue(has_rank or has_popularity, "Neither rank nor popularity field is present")
        self.assertTrue(has_rank != has_popularity, "Both rank and popularity fields are present. Only one value should be present")
        
        # If rank is present, verify it's one of the rank values
        if has_rank:
            self.assertIn(updated_doc['score_map.rank'], rank_values, 
                         f"Rank value {updated_doc['score_map.rank']} is not in expected values {rank_values}")
        
        # If popularity is present, verify it's one of the popularity values
        if has_popularity:
            self.assertIn(updated_doc['score_map.popularity'], popularity_values,
                         f"Popularity value {updated_doc['score_map.popularity']} is not in expected values {popularity_values}")
        
        # Verify original fields are still intact
        self.assertEqual(updated_doc['tensor_field'], 'concurrent update test')
        self.assertEqual(updated_doc['description'], 'This document will be updated by multiple threads')
        self.assertEqual(updated_doc['int_field'], 100)
        self.assertEqual(updated_doc['float_field'], 100.0)
        
        # Test search with score modifiers using the updated field (whichever is present)
        base_search_result = self.client.index(self.text_index_name).search("concurrent update")
        base_score = base_search_result["hits"][0]["_score"]

        if has_rank:
            search_result = self.client.index(self.text_index_name).search("concurrent update", score_modifiers={
                "add_to_score": [{"field_name": "score_map.rank", "weight": 1}]
            })
            
            self.assertTrue(len(search_result["hits"]) > 0, "No search results found")
            hit = search_result["hits"][0]
            self.assertAlmostEqual(hit["_score"], base_score + 1*updated_doc['score_map.rank'], places = 5)
            
        if has_popularity:
            search_result = self.client.index(self.text_index_name).search("concurrent update", score_modifiers={
                "add_to_score": [{"field_name": "score_map.popularity", "weight": 1}]
            })
            
            self.assertTrue(len(search_result["hits"]) > 0, "No search results found")
            hit = search_result["hits"][0]
            self.assertAlmostEqual(hit["_score"], base_score + 1*updated_doc['score_map.popularity'], places = 5)

    def test_update_document_with_changes_in_score_modifiers(self):
        """Test that score modifiers are correctly updated during partial document updates.
        
        This test verifies that:
        1. Score modifiers are properly updated when numeric fields are modified
        2. New numeric fields are correctly added to score modifiers
        3. The updated score modifiers affect search results as expected
        """
        # First add a document to update
        """Test updating a document with new fields and updating existing fields."""

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

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents = text_docs, mappings = mappings, tensor_fields = tensor_fields)

        self.assertFalse(add_docs_response["errors"])

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

