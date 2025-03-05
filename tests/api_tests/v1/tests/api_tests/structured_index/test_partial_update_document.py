import uuid
import random
import threading
import uuid

import numpy as np
import requests
from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestStructuredUpdateDocuments(MarqoTestCase):
    update_doc_index_name = "update_doc_api_test_index" + str(uuid.uuid4()).replace('-', '')
    large_score_modifier_index_name = ("update_doc_api_test_score_modifier_index" +
                                        str(uuid.uuid4()).replace('-', ''))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.update_doc_index_name,
                "type": "structured",
                "model": "random/small",
                "allFields": [
                    {"name": "text_field", "type": "text"},
                    {"name": "text_field_filter", "type": "text", "features": ["filter"]},
                    {"name": "text_field_lexical", "type": "text", "features": ["lexical_search"]},
                    {"name": "text_field_add", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_tensor", "type": "text"}, 
                    {"name": "int_field", "type": "int"},
                    {"name": "int_field_filter", "type": "int", "features": ["filter"]},
                    {"name": "int_field_score_modifier", "type": "int", "features": ["score_modifier"]},
                    {"name": "float_field", "type": "float"},
                    {"name": "float_field_filter", "type": "float", "features": ["filter"]},
                    {"name": "float_field_score_modifier", "type": "float", "features": ["score_modifier"]},
                    {"name": "bool_field_filter", "type": "bool", "features": ["filter"]},
                    {"name": "image_pointer_field", "type": "image_pointer"},
                    {"name": "dependent_field_1", "type": "text"},
                    {"name": "dependent_field_2", "type": "text"},
                    {"name": "multi_modal_field", "type": "multimodal_combination",
                     "dependentFields": {"dependent_field_1": 1.0, "dependent_field_2": 1.0}},
                    {"name": "array_text_field", "type": "array<text>", "features": ["filter"]},
                    {"name": "array_int_field", "type": "array<int>", "features": ["filter"]},
                ],
                "tensorFields": ["text_field_tensor", "multi_modal_field"],  # Specified as tensor fields
            },
            {
                "indexName": cls.large_score_modifier_index_name,
                "type": "structured",
                "model": "random/small",
                "allFields": [{"name": f"float_field_{i}", "type": "float", "features":
                    ["score_modifier", "filter"]} for i in range(100)] +
                [{"name": "text_field_tensor", "type": "text"}],
                "tensorFields": ["text_field_tensor"],
            }
        ]
        )

        cls.indexes_to_delete = [cls.update_doc_index_name, cls.large_score_modifier_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def set_up_for_text_field_test(self):
        """A helper function to set up the index to test the update document feature for text fields with
        different features."""
        original_doc = {
            "text_field": "text field",
            "text_field_filter": "text field filter",
            "text_field_lexical": "text field lexical",
            "text_field_tensor": "text field tensor",
            "_id": "1"
        }
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

    def set_up_for_int_field_test(self):
        """A helper function to set up the index to test the update document feature for int fields with
        different features."""
        original_doc = {
            "int_field": 1,
            "int_field_filter": 2,
            "int_field_score_modifier": 3,
            "text_field_tensor": "text field tensor",
            "_id": "1"
        }
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

    def set_up_for_float_field_test(self):
        """A helper function to set up the index to test the update document feature for float fields with
        different features."""
        original_doc = {
            "float_field": 1.1,
            "float_field_filter": 2.2,
            "float_field_score_modifier": 3.3,
            "text_field_tensor": "text field tensor",
            "_id": "1"
        }
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

    def test_update_text_field_filter(self):
        self.set_up_for_text_field_test()

        updated_doc = {
            "text_field_filter": "updated text field filter",
            "_id": "1"
        }
        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])

        # Assert that the text_field_filter has been updated as expected
        self.assertEqual("updated text field filter", updated_doc["text_field_filter"])

        # Perform a search to verify the document can be found with the updated filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="text_field_filter:(updated text field filter)"
        )
        self.assertEqual(1, len(search_result["hits"]))

        # Ensure the document cannot be found using the old text field filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="text_field_filter:(text field filter)"
        )
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_text_field_lexical(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_lexical": "search me please",
            "_id": "1"
        }
        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])

        # Assert that the text_field_lexical has been updated as expected
        self.assertEqual("search me please", updated_doc["text_field_lexical"])

        # Perform a lexical search to verify the document can be found with the updated text
        lexical_search_result = self.client.index(self.update_doc_index_name).search(
            q="search me please",
            searchable_attributes=["text_field_lexical"],
            search_method="LEXICAL"
        )
        self.assertEqual(1, len(lexical_search_result["hits"]))

        # Ensure the document cannot be found using the old text field lexical value
        lexical_search_result = self.client.index(self.update_doc_index_name).search(
            q="text field lexical",
            searchable_attributes=["text_field_lexical"],
            search_method="LEXICAL"
        )
        self.assertEqual(0, len(lexical_search_result["hits"]))

    def test_update_text_field_tensor(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_tensor": "I can't be updated",
            "_id": "1"
        }
        # Attempt to use the API client to update the document, expecting an error
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Assert that the update operation resulted in an error
        self.assertEqual(True, r["errors"])

        # Check that the error message includes a specific mention of the tensor field restriction
        self.assertIn("as this is a tensor field", r["items"][0]["error"])

    def test_update_text_field_add(self):
        """Ensure we can add a field to an indexing schema using the update document feature."""
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_add": "I am a new field",
            "_id": "1"
        }
        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual("I am a new field", updated_doc["text_field_add"])

        # Perform a lexical search to verify the document can be found with the added field
        lexical_search_result = self.client.index(self.update_doc_index_name).search(
            q="I am a new field",
            searchable_attributes=["text_field_add"],  # Assuming you meant to search the new field
            search_method="LEXICAL"
        )
        self.assertEqual(1, len(lexical_search_result["hits"]))

        # Perform a filter search to further verify the document with the new field
        filter_search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="text_field_add:(I am a new field)"
        )
        self.assertEqual(1, len(filter_search_result["hits"]))

    def test_update_int_field(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field": 11,
            "_id": "1"
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(11, updated_doc["int_field"])

    def test_update_int_field_filter(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field_filter": 22,
            "_id": "1"
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(22, updated_doc["int_field_filter"])

        # Perform a search to verify the document can be found with the updated filter value
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="int_field_filter:(22)"
        )
        self.assertEqual(1, len(search_result["hits"]))

        # Ensure the document cannot be found using the old integer value as a filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="int_field_filter:(2)"
        )
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_int_field_score_modifier(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field_score_modifier": 33,
            "_id": "1"
        }

        # Assuming the API has a way to specify score modifiers directly in the search request
        score_modifier = {
            "add_to_score": [{"field_name": "int_field_score_modifier", "weight": 1}]
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(33, updated_doc["int_field_score_modifier"])

        # Perform a search with the score modifier
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            score_modifiers=score_modifier  # This assumes the client and API support a similar structure
        )
        modified_score = search_result["hits"][0]["_score"]
        self.assertTrue(33 <= modified_score <= 34, f"Modified score is {modified_score}")

    def test_update_float_field(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field": 11.1,
            "_id": "1"
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(11.1, updated_doc["float_field"])

    def test_update_float_field_filter(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field_filter": 22.2,
            "_id": "1"
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(22.2, updated_doc["float_field_filter"])

        # Perform a search to verify the document can be found with the updated filter value
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="float_field_filter:(22.2)"
        )
        self.assertEqual(1, len(search_result["hits"]))

        # Ensure the document cannot be found using the old float value as a filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="float_field_filter:(2.2)"
        )
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_float_field_score_modifier(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field_score_modifier": 33.3,
            "_id": "1"
        }

        # Assuming the API has a way to specify score modifiers directly in the search request
        score_modifier = {
            "add_to_score": [{"field_name": "float_field_score_modifier", "weight": 1}]
        }

        # Use the API client to update the document
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document using the API client
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(33.3, updated_doc["float_field_score_modifier"])

        # Perform a search with the score modifier
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            score_modifiers=score_modifier  # This assumes the client and API support a similar structure
        )
        modified_score = search_result["hits"][0]["_score"]
        self.assertTrue(33.3 <= modified_score <= 34.3, f"Modified score is {modified_score}")

    def test_update_bool_field_filter(self):
        # Setup: Add the original document
        original_doc = {
            "bool_field_filter": True,
            "text_field_tensor": "search me",
            "_id": "1"
        }
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        # Assuming a method exists to verify the number of documents; adjust as necessary
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        # Update the document
        updated_doc = {
            "bool_field_filter": False,
            "_id": "1"
        }
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Verify the update
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(False, updated_doc["bool_field_filter"])

        # Search to verify the document with the updated boolean field
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="bool_field_filter:(false)"
        )
        self.assertEqual(1, len(search_result["hits"]))

        # Ensure the document cannot be found using the old boolean value as a filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="bool_field_filter:(true)"
        )
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_image_pointer_field(self):
        """Test that we can update the image_pointer_field in a document.

        Note: We can only update an image pointer field when it is not a tensor field."""
        original_doc = {
            "image_pointer_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
            "text_field_tensor": "search me",
            "_id": "1"
        }
        # Add the original document to the index
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        # Verify the document has been added
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        updated_doc = {
            "image_pointer_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            "_id": "1"
        }
        # Update the document's image pointer field
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document to verify the update
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual("https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                         updated_doc["image_pointer_field"])

    def test_update_multimodal_dependent_field(self):
        """Ensure that we CAN NOT update a multimodal dependent field."""
        original_doc = {
            "dependent_field_1": "dependent field 1",
            "dependent_field_2": "dependent field 2",
            "_id": "1"
        }
        # Add the original document to the index
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        # Verify the document has been added
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        updated_doc = {
            "dependent_field_1": "updated dependent field 1",
            "_id": "1"
        }
        # Attempt to update the document's multimodal dependent field
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Assuming the response `r` contains error details in a specific structure
        self.assertEqual(True, r["errors"])
        self.assertIn("dependent field", r["items"][0]["error"])

    def test_update_array_text_field_filter(self):
        original_doc = {
            "array_text_field": ["text1", "text2"],
            "text_field_tensor": "search me",
            "_id": "1"
        }
        # Add the original document to the index
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_doc])
        # Verify the document has been added
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        updated_doc = {
            "array_text_field": ["text3", "text4"],
            "_id": "1"
        }
        # Update the document's array of text fields
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document to verify the update
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])
        self.assertEqual(["text3", "text4"], updated_doc["array_text_field"])

        # Perform a search to verify the document can be found with the updated array text field
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="array_text_field:(text3)"
        )
        self.assertEqual(1, len(search_result["hits"]))

        # Ensure the document cannot be found using the old array text values as a filter
        search_result = self.client.index(self.update_doc_index_name).search(
            q="test",
            filter_string="array_text_field:(text1)"
        )
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_a_document_without_id(self):
        """Test attempting to update a document without providing an '_id' field."""
        updated_doc = {
            "text_field": "updated text field"
        }
        # Attempt to update a document without specifying an '_id'
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Check for errors indicating the missing '_id' field
        self.assertEqual(True, r["errors"])
        self.assertIn("'_id' is a required field but it does not exist", r["items"][0]["error"])
        self.assertEqual(400, r["items"][0]["status"])
        # Verify no documents have been inadvertently added or modified in the index
        self.assertEqual(0, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

    def test_update_multiple_fields_simultaneously(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "_id": "1",
            "text_field": "updated text field multi",
            "int_field_filter": 222,
            "float_field_score_modifier": 33.33,
            "bool_field_filter": True
        }
        # Use the API client to update the document with multiple fields
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Retrieve the updated document to verify the updates
        updated_doc = self.client.index(self.update_doc_index_name).get_document(updated_doc["_id"])

        # Assert that each field has been updated as expected
        self.assertEqual("updated text field multi", updated_doc["text_field"])
        self.assertEqual(222, updated_doc["int_field_filter"])
        self.assertEqual(33.33, updated_doc["float_field_score_modifier"])
        self.assertEqual(True, updated_doc["bool_field_filter"])

    def test_update_non_existent_field(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "_id": "1",
            "non_existent_field": "some value"  # Attempting to update a field that does not exist
        }
        # Attempt to update the document with a non-existent field
        r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        # Check for errors indicating the non-existent field
        self.assertEqual(True, r["errors"])
        self.assertIn("Invalid field name", r["items"][0]["error"])
        self.assertEqual(400, r["items"][0]["status"])

    def test_update_with_incorrect_field_value(self):
        self.set_up_for_text_field_test()

        test_cases = [
            ({"int_field_filter": "should be an integer"}, True, "This should be an integer"),
            ({"_id": 1}, True, "_id field should be a string"),
            ({"text_field": 1}, True, "This should be a string"),
            ({"bool_field_filter": "True"}, True, "This should be a boolean"),
            ({"float_field_score_modifier": "1.34"}, True, "This should be a float"),
            ({"array_text_field": "should be a list"}, True, "This should be a list"),
            ({"array_int_field": "should be a list"}, True, "This should be a list"),
            ({"array_int_field": [1, "should be an integer", 3]}, True, "This should be a list of integers"),
            ({"array_text_field": ["string", 2, "string"]}, True, "This should be a list of strings"),
        ]

        for updated_doc, expected_error, msg in test_cases:
            if "_id" not in updated_doc:
                updated_doc["_id"] = "1"
            with self.subTest(f"{updated_doc} - {msg}"):
                # Use the API client to attempt to update the document
                r = self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])
                # Check for an error response
                self.assertEqual(expected_error, r["errors"], msg=f"Failed on test case: {msg}")
                if expected_error:
                    self.assertTrue(r["errors"], msg=f"Expected an error for test case: {msg}")
                    self.assertTrue(r["items"][0]["status"] >= 400,
                                    msg=f"Expected a client or server error status for test case: {msg}")

    def test_multi_threading_update(self):
        """Test that we can update documents in multiple threads."""
        original_document = {
            "text_field": "text field",
            "text_field_filter": "text field filter",
            "text_field_lexical": "text field lexical",
            "text_field_tensor": "text field tensor",
            "int_field": 1,
            "int_field_filter": 2,
            "int_field_score_modifier": 3,
            "float_field": 1.1,
            "float_field_filter": 2.2,
            "bool_field_filter": True,
            "_id": "1"
        }
        self.client.index(self.update_doc_index_name).add_documents(documents=[original_document])
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        def randomly_update_document(number_of_updates: int = 50):
            updating_fields_pools = {"text_field", "text_field_filter", "text_field_lexical",
                                     "int_field", "int_field_filter", "int_field_score_modifier", "float_field",
                                     "float_field_filter", "bool_field_filter"}

            for _ in range(number_of_updates):
                picked_fields = random.sample(updating_fields_pools, 3)
                updated_doc = {"_id": "1"}
                for picked_field in picked_fields:
                    if picked_field.startswith("text_field"):
                        updated_doc[picked_field] = "text field" + str(random.randint(1, 100))
                    elif picked_field.startswith("int_field"):
                        updated_doc[picked_field] = np.random.randint(1, 100)
                    elif picked_field.startswith("float_field"):
                        updated_doc[picked_field] = np.random.uniform(1, 100)
                    elif picked_field.startswith("bool_field"):
                        updated_doc[picked_field] = bool(random.getrandbits(1))

                self.client.index(self.update_doc_index_name).update_documents(documents=[updated_doc])

        number_of_threads = 10
        updates_per_thread = 50

        threads = [threading.Thread(target=randomly_update_document, args=(updates_per_thread,)) for _ in
                   range(number_of_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        updated_doc = self.client.index(self.update_doc_index_name).get_document("1")

        # Assertions to ensure the document is not broken after the update
        self.assertEqual(1, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])
        self.assertTrue(updated_doc["text_field"].startswith("text field"))
        self.assertTrue(updated_doc["text_field_filter"].startswith("text field"))
        self.assertTrue(updated_doc["text_field_lexical"].startswith("text field"))
        self.assertTrue(updated_doc["text_field_tensor"].startswith("text field"))
        self.assertTrue(1 <= updated_doc["int_field"] <= 100)
        self.assertTrue(1 <= updated_doc["int_field_filter"] <= 100)
        self.assertTrue(1 <= updated_doc["int_field_score_modifier"] <= 100)
        self.assertTrue(1 <= updated_doc["float_field"] <= 100)
        self.assertTrue(1 <= updated_doc["float_field_filter"] <= 100)
        self.assertTrue(isinstance(updated_doc["bool_field_filter"], bool))

    def test_multi_threading_update_for_large_score_modifier_fields(self):
        """Test concurrent updates to multiple fields, focusing on ensuring document integrity."""
        original_document = {f"float_field_{i}": float(i) for i in range(100)}
        original_document["text_field_tensor"] = "text field tensor"
        original_document["_id"] = "1"

        # Add the original document to the index
        self.client.index(self.large_score_modifier_index_name).add_documents(documents=[original_document])
        self.assertEqual(1, self.client.index(self.large_score_modifier_index_name).get_stats()["numberOfDocuments"])

        def randomly_update_document(number_of_updates: int = 50):
            updating_fields_pools = [f"float_field_{i}" for i in range(100)]

            for _ in range(number_of_updates):
                picked_fields = random.sample(updating_fields_pools, 10)

                updated_doc = {"_id": "1"}
                for picked_field in picked_fields:
                    updated_doc[picked_field] = np.random.uniform(1, 100)

                # Update the document with randomly chosen fields
                self.client.index(self.large_score_modifier_index_name).update_documents(documents=[updated_doc])

        number_of_threads = 10
        updates_per_thread = 50

        threads = [threading.Thread(target=randomly_update_document, args=(updates_per_thread,)) for _ in
                   range(number_of_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Retrieve the updated document to ensure it's not broken
        updated_doc = self.client.index(self.large_score_modifier_index_name).get_document("1")
        self.assertEqual(1, self.client.index(self.large_score_modifier_index_name).get_stats()["numberOfDocuments"])
        for i in range(100):
            self.assertTrue(1 <= updated_doc[f"float_field_{i}"] <= 100)

        # Let do a final update and do a score modifier search to ensure the document is not broken
        final_doc = {f"float_field_{i}": 1.0 for i in range(100)}
        final_doc["_id"] = "1"
        r = self.client.index(self.large_score_modifier_index_name).update_documents(documents=[final_doc])

        original_score = self.client.index(self.large_score_modifier_index_name).search(q="test")["hits"][0]["_score"]

        for i in range(100):
            score_modifiers ={
                "add_to_score": [{"field_name": f"float_field_{i}", "weight": "1"}]
            }

            modified_score = self.client.index(self.large_score_modifier_index_name).search(q="test",
                score_modifiers=score_modifiers)["hits"][0]["_score"]

            self.assertAlmostEqual(original_score + 1, modified_score, 1)

    def test_batch_update_document_requests(self):
        """Test the client_batch_size parameter for the update_documents method."""
        original_doc = [
            {
                "text_field": "text field",
                "text_field_filter": "text field filter",
                "text_field_lexical": "text field lexical",
                "text_field_tensor": "text field tensor",
                "int_field": 1,
                "int_field_filter": 2,
                "int_field_score_modifier": 3,
                "float_field": 1.1,
                "float_field_filter": 2.2,
                "bool_field_filter": True,
                "_id": f"{i}"
            } for i in range(100)
        ]
        self.client.index(self.update_doc_index_name).add_documents(documents=original_doc)
        self.assertEqual(100, self.client.index(self.update_doc_index_name).get_stats()["numberOfDocuments"])

        updated_doc = [{
            "_id": f"{i}",
            "text_field": f"updated text field {i}",
            "int_field_filter": int(i),
            "float_field_score_modifier": float(i),
        } for i in range(100)]

        r = self.client.index(self.update_doc_index_name).\
            update_documents(documents=updated_doc, client_batch_size=10)

        self.assertEqual(10, len(r)) # 10 batches
        self.assertTrue(all([len(batch["items"]) == 10 for batch in r]), True) # 10 items in each batch

        for i in range(100):
            document = self.client.index(self.update_doc_index_name).get_document(f"{i}")
            self.assertEqual(f"updated text field {i}", document["text_field"])
            self.assertEqual(int(i), document["int_field_filter"])
            self.assertEqual(float(i), document["float_field_score_modifier"])

    def test_incorrect_update_document_body(self):
        base_url = self.client_settings["url"]
        update_doc_url = f"{base_url}/indexes/{self.update_doc_index_name}/documents"

        cases = [
            ({"documents": {"_id": "1", "text_field": "updated text field"}}, "Documents is not a list"),
            ([{"_id": "1", "text_field": "updated text field"}], "Body is missing the 'documents' key")
        ]

        for bad_body, msg in cases:
            with self.subTest(f"{bad_body} - {msg}"):
                r = requests.patch(update_doc_url, json=bad_body)
                self.assertEqual(422, r.status_code)
                self.assertIn("'body', 'documents'", str(r.json()))

    def test_too_many_documents_exceeds_max_batch_size(self):
        """Test that the update_documents method throws an error when the number of documents
        exceeds the max batch size."""
        documents = [{"_id": str(i), "text_field": f"updated text field {i}"} for i in range(129)]

        with self.assertRaises(MarqoWebError) as e:
            self.client.index(self.update_doc_index_name).update_documents(documents=documents)

        self.assertIn("Number of docs in update_documents request (129) exceeds", str(e.exception))
