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
        update_doc = {
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
            }
        
        update_docs_response = self.client.index(self.text_index_name).update_documents([
            update_doc])

        assert update_docs_response["errors"] == False

        get_docs_response = self.client.index(self.text_index_name).get_document(document_id = '1')
        
        # Use the helper method to verify all fields
        self._verify_document_fields(get_docs_response, get_docs_response)

    def test_add_new_fields_with_update_documents_api(self):
        """Test that new fields can be added to a document using partial updates.
        
        This test verifies that a document can be initially created with minimal fields,
        then updated to add new fields of various types, and that these fields are
        correctly stored and retrievable.

        Note: this test only covers the fields that can be successfully added.
        """
        # First add a minimal document with just an ID
        initial_doc = [{
            '_id': 'minimal_doc',
            'tensor_field': 'initial content'  # Need at least one field for indexing
        }]
        
        # Add the minimal document
        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=initial_doc,
            tensor_fields=['tensor_field']
        )
        self.assertFalse(add_docs_response["errors"])
        
        # Now update the document with new fields of various types. 
        # We won't be adding new lexical fields, as that is not supported. 
        update_doc = {
            '_id': 'minimal_doc',
            'new_int_field': 42,
            'new_float_field': 3.14159,
            'new_bool_field': True,
            'new_int_map': {
                'key1': 100,
                'key2': 200
            },
            'new_float_map': {
                'pi': 3.14,
                'e': 2.718
            }
        }
        update_docs_response = self.client.index(self.text_index_name).update_documents([update_doc])
        
        self.assertFalse(update_docs_response["errors"])
        
        # Get the document back and verify all fields exist with correct values
        get_docs_response = self.client.index(self.text_index_name).get_document(document_id='minimal_doc')
        
        # Verify original field
        self.assertEqual(get_docs_response['tensor_field'], 'initial content')
        
        # Check all fields in the update document
        self._verify_document_fields(update_doc, get_docs_response)

    def test_add_new_tensor_field_with_update_documents_api(self):
        """Test that adding new tensor fields via update_documents fails.
        
        This test verifies that attempting to add new tensor fields during an update operation
        results in appropriate error responses, as these field types cannot be added after 
        initial document creation.
        """
        # First add a minimal document with just an ID and a tensor field
        initial_doc = [{
            '_id': 'tensor_update_doc',
            'tensor_field': 'initial content'  # Need at least one field for indexing
        }]
        
        # Add the minimal document
        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=initial_doc,
            tensor_fields=['tensor_field']
        )
        self.assertFalse(add_docs_response["errors"])
        
        # Try to update with a new tensor field
        update_docs_response = self.client.index(self.text_index_name).update_documents([{
            '_id': 'tensor_update_doc',
            'new_tensor_field': 'This should fail'
        }])
        
        # Verify that errors were returned
        self.assertTrue(update_docs_response["errors"], 
                       "Expected error when adding new tensor field but got success")
        
        # Check the error details
        error_details = update_docs_response["items"][0]
        self.assertEqual(error_details["status"], 400, 
                        "Expected status code 400 for tensor field but got {error_details['status']}")

        # Assert the error message
        self.assertIn("new_tensor_field of type str does not exist in the original document. "
                      "Marqo does not support adding new lexical fields in partial updates"
                      , error_details["error"])

    def test_add_new_string_array_with_update_documents_api(self):
        """Test that adding new string arrays via update_documents fails.
        
        This test verifies that attempting to add new string arrays during an update operation
        results in appropriate error responses, as these field types cannot be added after 
        initial document creation.
        """
        # First add a minimal document with just an ID and a tensor field
        initial_doc = [{
            '_id': 'string_array_update_doc',
            'tensor_field': 'initial content'  # Need at least one field for indexing
        }]
        
        # Add the minimal document
        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=initial_doc,
            tensor_fields=['tensor_field']
        )
        self.assertFalse(add_docs_response["errors"])
        
        # Try to update with a new string array
        update_docs_response = self.client.index(self.text_index_name).update_documents([{
            '_id': 'string_array_update_doc',
            'new_string_array': ['item1', 'item2']
        }])
        
        # Verify that errors were returned
        self.assertTrue(update_docs_response["errors"], 
                       "Expected error when adding new string array but got success")
        
        # Check the error details
        error_details = update_docs_response["items"][0]
        self.assertEqual(error_details["status"], 400, 
                        "Expected status code 400 for string array but got {error_details['status']}")

        # Assert the error message
        self.assertIn("Unstructured index updates only support updating existing string array fields"
                      , error_details["error"])

    def test_add_new_string_field_with_update_documents_api(self):
        """Test that adding new string fields via update_documents fails.
        
        This test verifies that attempting to add new string fields during an update operation
        results in appropriate error responses, as these field types cannot be added after 
        initial document creation.
        """
        # First add a minimal document with just an ID and a tensor field
        initial_doc = [{
            '_id': 'string_field_update_doc',
            'tensor_field': 'initial content'  # Need at least one field for indexing
        }]
        
        # Add the minimal document
        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=initial_doc,
            tensor_fields=['tensor_field']
        )
        self.assertFalse(add_docs_response["errors"])
        
        # Try to update with a new string field
        update_docs_response = self.client.index(self.text_index_name).update_documents([{
            '_id': 'string_field_update_doc',
            'new_string_field': 'This should also fail'
        }])
        
        # Verify that errors were returned
        self.assertTrue(update_docs_response["errors"], 
                       "Expected error when adding new string field but got success")
        
        # Check the error details
        error_details = update_docs_response["items"][0]
        self.assertEqual(error_details["status"], 400, 
                        "Expected status code 400 for string field but got {error_details['status']}")

        # Assert the error message
        self.assertIn("new_string_field of type str does not exist in the original document. "
                      "Marqo does not support adding new lexical fields in partial updates"
                      , error_details["error"])

    def test_update_document_and_change_field_type(self):
        """Test that changing field types during document updates fails with appropriate errors.
        
        This test verifies that attempting to change a field's type (e.g., float to int, int to string)
        during an update operation results in the expected error responses.
        """
        # First add a document with various field types
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

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents=text_docs, mappings=mappings, tensor_fields=tensor_fields)
        self.assertFalse(add_docs_response["errors"])

        # Define field type change scenarios to test
        type_change_scenarios = [
            {"field": "float_field", "new_value": 500, "original_type": "float", "new_type": "int"},
            {"field": "int_field", "new_value": 123.5, "original_type": "int", "new_type": "float"},
            {"field": "bool_field", "new_value": "true", "original_type": "bool", "new_type": "string"},
            {"field": "short_string_field", "new_value": 42, "original_type": "string", "new_type": "int"},
            {"field": "int_map", "new_value": {"a": 2.5}, "original_type": "int_map", "new_type": "float_map"},
            {"field": "float_map", "new_value": {"c": 3}, "original_type": "float_map", "new_type": "int_map"},
            {"field": "string_array", "new_value": 123, "original_type": "string_array", "new_type": "int"},
            # Map to scalar type changes
            {"field": "int_map", "new_value": 42, "original_type": "int_map", "new_type": "int"},
            {"field": "int_map", "new_value": 42.5, "original_type": "int_map", "new_type": "float"},
            {"field": "float_map", "new_value": 42, "original_type": "float_map", "new_type": "int"},
            {"field": "float_map", "new_value": 42.5, "original_type": "float_map", "new_type": "float"}
        ]

        # Test each type change scenario in a subtest
        for scenario in type_change_scenarios:
            with self.subTest(f"Changing {scenario['field']} from {scenario['original_type']} to {scenario['new_type']}"):
                update_payload = {
                    '_id': '1',
                    scenario['field']: scenario['new_value']
                }
                
                update_docs_response = self.client.index(self.text_index_name).update_documents([update_payload])

                # Verify the update failed with appropriate error
                self.assertTrue(update_docs_response["errors"])
                self.assertEqual(update_docs_response['items'][0]['status'], 400)
                if scenario['original_type'] == "bool":
                    self.assertIn("bool_field of type str does not exist in the original document. Marqo does not support adding new lexical fields in partial updates",
                                  update_docs_response['items'][0]['error'])
                else:
                    self.assertIn("Marqo vector store couldn't update the document. Please see",
                                  update_docs_response['items'][0]['message'])
                    self.assertIn("reference/api/documents/update-documents/#response",
                                  update_docs_response['items'][0]['message'])
                
    def test_concurrent_partial_update_requests_on_maps(self):
        """Test concurrent updates to different fields of the same document.
        
        This test verifies that:
        1. Multiple threads can update different fields of the same document concurrently
        2. Updates are applied, some of which fail due to concurrent updates, but the final state is consistent
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
                # We can't be sure that the response will be error-free due to concurrent updates. So we will not check that here
                print(f"[{timestamp}] Rank update {i+1} complete. Response: {r}")
                time.sleep(0.5)  # Small delay between updates

        def update_popularity_thread(index_name, popularity_values):
            for i, new_pop in enumerate(popularity_values):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Popularity update {i+1}/{len(popularity_values)}: Setting popularity to {new_pop}")
                r = self.client.index(index_name).update_documents([{'_id': '3', 'score_map': {'popularity': new_pop}}])
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                # We can't be sure that the response will be error-free due to concurrent updates. So we will not check that here
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

    def test_concurrent_partial_update_requests_on_numeric_fields(self):
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

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents=text_docs, mappings={},
                                                                                  tensor_fields=['tensor_field',
                                                                                                 'description'])
        self.assertFalse(add_docs_response["errors"])

        def update_rank_thread(index_name, rank_values):
            for i, new_rank in enumerate(rank_values):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Rank update {i + 1}/{len(rank_values)}: Setting rank to {new_rank}")
                r = self.client.index(index_name).update_documents([{'_id': '3', 'rank': new_rank}])
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Rank update {i + 1} complete. Response: {r}")
                self.assertFalse(r["errors"]) # Assert that there are no errors in the response
                time.sleep(0.5)  # Small delay between updates

        def update_popularity_thread(index_name, popularity_values):
            for i, new_pop in enumerate(popularity_values):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(
                    f"[{timestamp}] Popularity update {i + 1}/{len(popularity_values)}: Setting popularity to {new_pop}")
                r = self.client.index(index_name).update_documents([{'_id': '3', 'popularity': new_pop}])
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Popularity update {i + 1} complete. Response: {r}")
                self.assertFalse(r["errors"]) # Assert that there are no errors in the response
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
        has_rank = 'rank' in updated_doc
        has_popularity = 'popularity' in updated_doc

        # Either rank or popularity should be present, but not both
        self.assertTrue(has_rank and has_popularity, "Neither rank nor popularity field is present")

        # If rank is present, verify it's one of the rank values
        self.assertIn(updated_doc['rank'], rank_values,
                      f"Rank value {updated_doc['rank']} is not in expected values {rank_values}")

        # If popularity is present, verify it's one of the popularity values
        self.assertIn(updated_doc['popularity'], popularity_values,
                          f"Popularity value {updated_doc['popularity']} is not in expected values {popularity_values}")

        # Verify original fields are still intact
        self.assertEqual(updated_doc['tensor_field'], 'concurrent update test')
        self.assertEqual(updated_doc['description'], 'This document will be updated by multiple threads')
        self.assertEqual(updated_doc['int_field'], 100)
        self.assertEqual(updated_doc['float_field'], 100.0)

        # Test search with score modifiers using the updated field (whichever is present)
        base_search_result = self.client.index(self.text_index_name).search("concurrent update")
        base_score = base_search_result["hits"][0]["_score"]

        search_result = self.client.index(self.text_index_name).search("concurrent update", score_modifiers={
            "add_to_score": [{"field_name": "rank", "weight": 1}]
        })

        self.assertTrue(len(search_result["hits"]) > 0, "No search results found")
        hit = search_result["hits"][0]
        self.assertAlmostEqual(hit["_score"], base_score + 1 * updated_doc['rank'], places=5)

        search_result = self.client.index(self.text_index_name).search("concurrent update", score_modifiers={
            "add_to_score": [{"field_name": "popularity", "weight": 1}]
        })

        self.assertTrue(len(search_result["hits"]) > 0, "No search results found")
        hit = search_result["hits"][0]
        self.assertAlmostEqual(hit["_score"], base_score + 1 * updated_doc['popularity'], places=5)

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

        update_doc = {
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
        }

        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [update_doc]
        )

        self.assertFalse(update_docs_response["errors"])

        # Get the document to verify updates
        updated_doc = self.client.index(self.text_index_name).get_document(document_id='1')
        
        self._verify_document_fields(update_doc, updated_doc)

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

    def test_concurrent_mixed_add_documents_and_update_documents_requests(self):
        """Test concurrent updates using different API methods on the same document.

        This test verifies that:
        1. Multiple threads can update the same document using different API methods concurrently
        2. One thread uses update_documents to modify specific fields
        3. One thread uses add_documents to replace the entire document
        4. The document remains in a consistent state after all operations
        """
        # First add a document to update
        text_docs = [{
            '_id': '4',
            'tensor_field': 'mixed update test',
            'description': 'This document will be updated by multiple threads using different methods',
            'int_field': 100,
            'float_field': 100.0,
            'tags': ['initial', 'document'],
        }]

        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=text_docs,
            mappings={},
            tensor_fields=['tensor_field', 'description']
        )
        self.assertFalse(add_docs_response["errors"])

        def update_documents_thread(index_name):
            """Thread that updates specific fields using update_documents."""
            for i in range(10):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] update_documents thread - iteration {i + 1}/10: Updating int_field and metadata")
                r = self.client.index(index_name).update_documents([{
                    '_id': '4',
                    'int_field': 200 + i,
                    'tensor_field': f'update_documents replaced content {i + 1}',
                    'description': f'This document was updated in iteration {i + 1} inside an update_documents thread',
                }])
                self.assertTrue(r["errors"]) # We can be sure that the response will be error-free due to concurrent updates and the updates sent by the other add documents
                # thread
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] update_documents thread - iteration {i + 1} complete. Response: {r}")
                time.sleep(0.5)  # Delay between updates

        def add_documents_thread(index_name):
            """Thread that replaces the entire document using add_documents."""
            for i in range(10):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] add_documents thread - iteration {i + 1}/10: Replacing document")
                r = self.client.index(index_name).add_documents(
                    documents=[{
                        '_id': '4',
                        'tensor_field': f'add_documents replaced content {i + 1}',
                        'description': f'This document was replaced in iteration {i + 1} inside an add_documents thread',
                        'int_field': 300 + i,
                        'float_field': 300.0 + i,
                        'tags': ['replaced', f'iteration-{i + 1}'],
                    }],
                    tensor_fields=['tensor_field', 'description']
                )
                self.assertFalse(r["errors"]) # Assert that there are no errors in the response
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] add_documents thread - iteration {i + 1} complete. Response: {r}")
                time.sleep(0.5)  # Delay between updates

        # Create and start threads
        update_thread = threading.Thread(target=update_documents_thread, args=(self.text_index_name,))
        add_thread = threading.Thread(target=add_documents_thread, args=(self.text_index_name,))

        add_thread.start()
        update_thread.start()

        add_thread.join()
        update_thread.join()

        # Verify the final document state
        final_doc = self.client.index(self.text_index_name).get_document(document_id='4')

        # We can't predict exactly which thread's updates will be the final state
        # but we can verify that the document exists and has expected structure
        self.assertEqual(final_doc['_id'], '4')
        self.assertIn('tensor_field', final_doc)
        self.assertIn('description', final_doc)
        self.assertIn('int_field', final_doc)
        self.assertIn('float_field', final_doc)
        self.assertIn('This document was replaced in iteration 10 inside an add_documents thread', final_doc['description'])
        self.assertIn('add_documents replaced content 10', final_doc['tensor_field'])

    def test_partial_update_new_map_field(self):
        """
        Test the following scenario:
        1. Add a map with a partial update,
        2. Update it with another partial update, remove the existing keys
        3. Final state of the doc should only contain the Map sent in #2
        """
        text_docs = [
            {
                '_id': '4',
                'text': 'This is a test document',
                'rank': 100,
            }
        ]
        add_docs_response = self.client.index(self.text_index_name).add_documents(
            documents=text_docs,
            mappings={},
            tensor_fields=['text']
        )

        get_doc = self.client.index(self.text_index_name).get_document('4')
        self.assertEqual(get_doc['rank'], text_docs[0]['rank'])

        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [ {
                '_id': '4',
                'metadata': {'key1': 100.5}
            }]
        )
        self.assertFalse(update_docs_response['errors'])

        update_docs_response_2 = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '4',
                'metadata': {'key2': 100.5},
            }]
        )

        self.assertFalse(update_docs_response_2['errors'])

        get_docs_result = self.client.index(index_name=self.text_index_name).get_document(document_id='4')
        self.assertIsNone(get_docs_result.get('metadata.key1'))
        self.assertEqual(get_docs_result['metadata.key2'], 100.5)

        base_search_result = self.client.index(index_name=self.text_index_name).search("test document")
        base_score = base_search_result["hits"][0]["_score"]

        search_result_with_non_existent_score_modifier = self.client.index(index_name = self.text_index_name).search("test document", score_modifiers={
            "add_to_score": [{"field_name": "metadata.key1", "weight": 1}]
        })

        score_with_non_existent_score_modifier = search_result_with_non_existent_score_modifier["hits"][0]["_score"]
        # Since the score modifier should not exist after the map has been completely replaced, the scores should be the same
        self.assertAlmostEqual(score_with_non_existent_score_modifier, base_score, 5)

        search_with_existing_score_modifier = self.client.index(index_name = self.text_index_name).search("test document", score_modifiers={
            "add_to_score": [{"field_name": "metadata.key2", "weight": 1}]
        })

        score_with_existing_score_modifier = search_with_existing_score_modifier["hits"][0]["_score"]
        self.assertAlmostEqual(score_with_existing_score_modifier, base_score + 1*100.5, 5)


    def _verify_document_fields(self, update_doc, get_docs_response):
        """Verify that all fields in the update document are correctly stored in the retrieved document.
        
        This helper method checks each field in the update document against the retrieved document,
        handling both simple fields and nested map fields appropriately.
        
        Args:
            update_doc: The document used in the update operation
            get_docs_response: The document retrieved from the index after update
        """
        for field, expected_value in update_doc.items():
            if field == '_id':
                continue
            if isinstance(expected_value, dict):
                # Handle map fields
                for key, value in expected_value.items():
                    flattened_field = f"{field}.{key}"
                    self.assertEqual(
                        get_docs_response[flattened_field], 
                        value, 
                        f"Field {flattened_field} value mismatch: expected {value}, got {get_docs_response[flattened_field]}"
                    )
            else:
                # Handle simple fields
                self.assertEqual(
                    get_docs_response[field], 
                    expected_value,
                    f"Field {field} value mismatch: expected {expected_value}, got {get_docs_response[field]}"
                )
