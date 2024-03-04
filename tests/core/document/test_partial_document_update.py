import os
import random
import threading
from unittest import mock

import numpy as np

from marqo.api.exceptions import BadRequestError
from marqo.api.models.update_documents import UpdateDocumentsBodyParams
from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.api import update_documents
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from tests.marqo_test import MarqoTestCase


class TestUpdate(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        structured_index_request_1 = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name='text_field', type=FieldType.Text),
                FieldRequest(name='text_field_filter', type=FieldType.Text, features=[FieldFeature.Filter]),
                FieldRequest(name='text_field_lexical', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='text_field_add', type=FieldType.Text, features=[FieldFeature.Filter,
                                                                                   FieldFeature.LexicalSearch]),
                FieldRequest(name='text_field_tensor', type=FieldType.Text),  # A tensor field

                FieldRequest(name='int_field', type=FieldType.Int),
                FieldRequest(name='int_field_filter', type=FieldType.Int, features=[FieldFeature.Filter]),
                FieldRequest(name='int_field_score_modifier', type=FieldType.Int,
                             features=[FieldFeature.ScoreModifier]),

                FieldRequest(name='float_field', type=FieldType.Float),
                FieldRequest(name='float_field_filter', type=FieldType.Float, features=[FieldFeature.Filter]),
                FieldRequest(name='float_field_score_modifier', type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),

                FieldRequest(name="bool_field_filter", type=FieldType.Bool, features=[FieldFeature.Filter]),

                FieldRequest(name="image_pointer_field", type=FieldType.ImagePointer),

                FieldRequest(name="dependent_field_1", type=FieldType.Text),
                FieldRequest(name="dependent_field_2", type=FieldType.Text),
                FieldRequest(name="multi_modal_field", type=FieldType.MultimodalCombination,
                             dependent_fields={"dependent_field_1": 1.0, "dependent_field_2": 1.0}),

                FieldRequest(name="array_text_field", type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name="array_int_field", type=FieldType.ArrayInt, features=[FieldFeature.Filter]),
            ],
            tensor_fields=['text_field_tensor', "multi_modal_field"],
            model=Model(name="random/small")
        )

        large_score_modifiers_index_fields = [
            FieldRequest(name=f"float_field_{i}", type=FieldType.Float,
                         features=[FieldFeature.Filter, FieldFeature.ScoreModifier]) for i in range(100)
        ]
        large_score_modifiers_index_fields.append(
            FieldRequest(name="text_field_tensor", type=FieldType.Text)
        )

        large_score_modifiers_index_request = cls.structured_marqo_index_request(
            fields=large_score_modifiers_index_fields,
            tensor_fields=["text_field_tensor"],
            model=Model(name="random/small")

        )

        test_unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name="random/small"),
        )

        cls.indexes = cls.create_indexes([
            structured_index_request_1,
            large_score_modifiers_index_request,
            test_unstructured_index_request
        ])

        cls.structured_index_name = structured_index_request_1.name
        cls.large_score_modifier_index_name = large_score_modifiers_index_request.name
        cls.test_unstructured_index_name = test_unstructured_index_request.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

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
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

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
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

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
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

    def test_update_text_field(self):
        self.set_up_for_text_field_test()

        updated_doc = {
            "text_field": "updated text field",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual("updated text field", updated_doc["text_field"])

    def test_update_text_field_filter(self):
        self.set_up_for_text_field_test()

        updated_doc = {
            "text_field_filter": "updated text field filter",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual("updated text field filter", updated_doc["text_field_filter"])

        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="text_field_filter:(updated text field filter)")
        self.assertEqual(1, len(search_result["hits"]))
        # Ensure we can't filter on the old text
        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="text_field_filter:(text field filter)")
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_text_field_lexical(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_lexical": "search me please",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual("search me please", updated_doc["text_field_lexical"])
        lexical_search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                                     search_method="LEXICAL", text="search me please",
                                                     searchable_attributes=["text_field_lexical"])
        self.assertEqual(1, len(lexical_search_result["hits"]))
        # Ensure we can't lexical search on the old text
        lexical_search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                                     search_method="LEXICAL", text="text field lexical",
                                                     searchable_attributes=["text_field_lexical"])
        self.assertEqual(0, len(lexical_search_result["hits"]))

    def test_update_text_field_tensor(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_tensor": "I can't be updated",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        self.assertEqual(True, r["errors"])
        self.assertIn("as this is a tensor field", r["items"][0]["error"])

    def test_update_text_field_add(self):
        """Ensure we can add a field to an indexing schema using the update document feature."""
        self.set_up_for_text_field_test()
        updated_doc = {
            "text_field_add": "I am a new field",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual("I am a new field", updated_doc["text_field_add"])

        lexical_search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                                     search_method="LEXICAL", text="I am a new field",
                                                     searchable_attributes=["text_field_lexical"])
        self.assertEqual(1, len(lexical_search_result["hits"]))

        filter_search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                                    text="test", filter="text_field_add:(I am a new field)")
        self.assertEqual(1, len(filter_search_result["hits"]))

    def test_update_int_field(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field": 11,
            "_id": "1"
        }

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(11, updated_doc["int_field"])

    def test_update_int_field_filter(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field_filter": 22,
            "_id": "1"
        }

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(22, updated_doc["int_field_filter"])

        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="int_field_filter:(22)")
        self.assertEqual(1, len(search_result["hits"]))
        # Ensure we can't filter on the old int
        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="int_field_filter:(2)")
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_int_field_score_modifier(self):
        self.set_up_for_int_field_test()
        updated_doc = {
            "int_field_score_modifier": 33,
            "_id": "1"
        }

        score_modifier = ScoreModifier(**{
            "add_to_score": [{"field_name": "int_field_score_modifier", "weight": "1"}]
        })

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(33, updated_doc["int_field_score_modifier"])

        modified_score = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                              text="test", score_modifiers=score_modifier)["hits"][0]["_score"]
        self.assertTrue(33 <= modified_score <= 34, f"Modified score is {modified_score}")

    def test_update_float_field(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field": 11.1,
            "_id": "1"
        }

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(11.1, updated_doc["float_field"])

    def test_update_float_field_filter(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field_filter": 22.2,
            "_id": "1"
        }

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(22.2, updated_doc["float_field_filter"])

        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="float_field_filter:(22.2)")
        self.assertEqual(1, len(search_result["hits"]))
        # Ensure we can't filter on the old float
        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="float_field_filter:(2.2)")
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_float_field_score_modifier(self):
        self.set_up_for_float_field_test()
        updated_doc = {
            "float_field_score_modifier": 33.3,
            "_id": "1"
        }

        score_modifier = ScoreModifier(**{
            "add_to_score": [{"field_name": "float_field_score_modifier", "weight": "1"}]
        })

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
        self.assertEqual(33.3, updated_doc["float_field_score_modifier"])

        modified_score = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                              text="test", score_modifiers=score_modifier)["hits"][0]["_score"]
        self.assertTrue(33.3 <= modified_score <= 34.3, f"Modified score is {modified_score}")

    def test_update_bool_field_filter(self):
        original_doc = {
            "bool_field_filter": True,
            "text_field_tensor": "search me",
            "_id": "1"
        }
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        updated_doc = {
            "bool_field_filter": False,
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual(False, updated_doc["bool_field_filter"])

        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="bool_field_filter:(false)")
        self.assertEqual(1, len(search_result["hits"]))
        # Ensure we can't filter on the old bool value
        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="bool_field_filter:(true)")
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_image_pointer_field(self):
        """Test that we can update the image_pointer_field in a document.

        Note: We can only update an image pointer field when it is not a tensor field."""
        original_doc = {
            "image_pointer_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg",
            "text_field_tensor": "search me",
            "_id": "1"
        }
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        updated_doc = {
            "image_pointer_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual("https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
                         updated_doc["image_pointer_field"])

    def test_update_multimodal_dependent_field(self):
        """Ensure that we CAN NOT update a multimodal dependent field."""
        original_doc = {
            "dependent_field_1": "dependent field 1",
            "dependent_field_2": "dependent field 2",
            "_id": "1"
        }
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        updated_doc = {
            "dependent_field_1": "updated dependent field 1",
            "_id": "1"
        }

        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        self.assertEqual(True, r["errors"])
        self.assertIn("dependent field", r["items"][0]["error"])

    def test_update_array_text_field_filter(self):
        original_doc = {
            "array_text_field": ["text1", "text2"],
            "text_field_tensor": "search me",
            "_id": "1"
        }
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_doc]
        ))
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        updated_doc = {
            "array_text_field": ["text3", "text4"],
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual(["text3", "text4"], updated_doc["array_text_field"])

        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="array_text_field:(text3)")
        self.assertEqual(1, len(search_result["hits"]))
        # Ensure we can't filter on the old array<text> value
        search_result = tensor_search.search(config=self.config, index_name=self.structured_index_name,
                                             text="test", filter="array_text_field:(text1)")
        self.assertEqual(0, len(search_result["hits"]))

    def test_update_a_document_that_does_not_exist(self):
        """"""
        updated_doc = {
            "text_field": "updated text field",
            "_id": "1"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        self.assertEqual(True, r["errors"])
        self.assertIn("Document does not exist in the index", r["items"][0]["error"])
        self.assertEqual(404, r["items"][0]["status"])
        self.assertEqual(0, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

    def test_update_a_document_without_id(self):
        updated_doc = {
            "text_field": "updated text field"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        self.assertEqual(True, r["errors"])
        self.assertIn("'_id' is a required field but it does not exist", r["items"][0]["error"])
        self.assertEqual(400, r["items"][0]["status"])
        self.assertEqual(0, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

    def test_update_multiple_fields_simultaneously(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "_id": "1",
            "text_field": "updated text field multi",
            "int_field_filter": 222,
            "float_field_score_modifier": 33.33,
            "bool_field_filter": True
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])

        self.assertEqual("updated text field multi", updated_doc["text_field"])
        self.assertEqual(222, updated_doc["int_field_filter"])
        self.assertEqual(33.33, updated_doc["float_field_score_modifier"])
        self.assertEqual(True, updated_doc["bool_field_filter"])

    def test_update_non_existent_field(self):
        self.set_up_for_text_field_test()
        updated_doc = {
            "_id": "1",
            "non_existent_field": "some value"
        }
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                             index_name=self.structured_index_name, marqo_config=self.config)
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
                r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                                     index_name=self.structured_index_name, marqo_config=self.config)
                self.assertEqual(expected_error, r["errors"])
                print(r)
                if expected_error:
                    self.assertEqual(True, r["errors"])
                    self.assertTrue(r["items"][0]["status"] >= 400)

    def test_multi_threading_update(self):
        """Test that we can update documents in multiple threads.

        Note: We are not able to serialise the requests so the order of the updates is not guaranteed. We are only
        ensuring that the documents are not broken after the update."""
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

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.structured_index_name,
            docs=[original_document]
        ))

        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        def randomly_update_document(number_of_updates: int = 50) -> None:
            updating_fields_pools = {"text_field", "text_field_filter", "text_field_lexical", "text_field_tensor",
                                     "int_field", "int_field_filter", "int_field_score_modifier", "float_field",
                                     "float_field_filter", "bool_field_filter"}

            for _ in range(number_of_updates):
                picked_fields = random.sample(updating_fields_pools, 3)

                updated_doc = {
                    "_id": "1"
                }
                for picked_field in picked_fields:

                    if picked_field.startswith("text_field"):
                        updated_doc[picked_field] = "text field" + str(random.randint(1, 100))
                    elif picked_field.startswith("int_field"):
                        updated_doc[picked_field] = np.random.randint(1, 100)
                    elif picked_field.startswith("float_field"):
                        updated_doc[picked_field] = np.random.uniform(1, 100)
                    elif picked_field.startswith("bool_field"):
                        updated_doc[picked_field] = bool(random.getrandbits(1))
                    else:
                        raise ValueError(f"Invalid field name {picked_field}")

                r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                                     index_name=self.structured_index_name, marqo_config=self.config)

        number_of_threads = 10
        updates_per_thread = 50

        threads = [threading.Thread(target=randomly_update_document(updates_per_thread),
                                    args=[updates_per_thread]) for _ in range(number_of_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, "1")

        # We only want to ensure the document is not broken after the update
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_index_name).number_of_documents)

        self.assertEqual(updated_doc["text_field"].startswith("text field"), True)
        self.assertEqual(updated_doc["text_field_filter"].startswith("text field"), True)
        self.assertEqual(updated_doc["text_field_lexical"].startswith("text field"), True)
        self.assertEqual(updated_doc["text_field_tensor"].startswith("text field"), True)
        self.assertTrue(1 <= updated_doc["int_field"] <= 100)
        self.assertTrue(1 <= updated_doc["int_field_filter"] <= 100)
        self.assertTrue(1 <= updated_doc["int_field_score_modifier"] <= 100)
        self.assertTrue(1 <= updated_doc["float_field"] <= 100)
        self.assertTrue(1 <= updated_doc["float_field_filter"] <= 100)
        self.assertTrue(isinstance(updated_doc["bool_field_filter"], bool))

    def test_multi_threading_update_for_large_score_modifier_fields(self):
        """Test that we can update multiple score modifier fields in multiple threads.

        Note: We are not able to serialise the requests so the order of the updates is not guaranteed. We are only
        ensuring that the documents are not broken after the update.

        The reason that we are focusing on the score modifier fields is that all the score modifiers are stored in
        a single vespa tensor field, and we are updating the tensor while we update a score modifier field.
        We want to ensure that the score modifier fields are updated correctly and the tensor field is not broken."""

        original_document = {f"float_field_{i}": float(i) for i in range(100)}
        original_document["text_field_tensor"] = "text field tensor"
        original_document["_id"] = "1"

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.large_score_modifier_index_name,
            docs=[original_document]
        ))

        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.large_score_modifier_index_name) \
                         .number_of_documents)

        def randomly_update_document(number_of_updates: int = 50) -> None:
            updating_fields_pools = [f"float_field_{i}" for i in range(100)]

            for _ in range(number_of_updates):
                picked_fields = random.sample(updating_fields_pools, 10)

                updated_doc = {
                    "_id": "1"
                }
                for picked_field in picked_fields:
                    updated_doc[picked_field] = np.random.uniform(1, 100)

                r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                                     index_name=self.large_score_modifier_index_name, marqo_config=self.config)

        number_of_threads = 10
        updates_per_thread = 50

        threads = [threading.Thread(target=randomly_update_document(updates_per_thread),
                                    args=[updates_per_thread]) for _ in range(number_of_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        updated_doc = tensor_search.get_document_by_id(self.config, self.large_score_modifier_index_name, "1")

        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.large_score_modifier_index_name). \
                         number_of_documents)
        for i in range(100):
            self.assertTrue(1 <= updated_doc[f"float_field_{i}"] <= 100)

        # Let do a final update and do a score modifier search to ensure the document is not broken
        final_doc = {f"float_field_{i}": 1.0 for i in range(100)}
        final_doc["_id"] = "1"
        r = update_documents(body=UpdateDocumentsBodyParams(documents=[final_doc]),
                             index_name=self.large_score_modifier_index_name,
                             marqo_config=self.config)

        original_score = tensor_search.search(config=self.config, index_name=self.large_score_modifier_index_name,
                                              text="test")["hits"][0]["_score"]

        for i in range(100):
            score_modifiers = ScoreModifier(**{
                "add_to_score": [{"field_name": f"float_field_{i}", "weight": "1"}]
            })

            modified_score = tensor_search.search(config=self.config, index_name=self.large_score_modifier_index_name,
                                                  text="test", score_modifiers=score_modifiers)["hits"][0]["_score"]

            self.assertAlmostEqual(original_score + 1, modified_score, 1)

    def test_proper_error_raised_if_received_too_many_documents(self):
        with self.assertRaises(BadRequestError) as cm:
            r = update_documents(body=UpdateDocumentsBodyParams(documents=[{"_id": "1"}] * 129),
                                 index_name=self.structured_index_name, marqo_config=self.config)

        # The same request (size) should work if the max batch size is increased
        with mock.patch.dict(os.environ, {"MARQO_MAX_DOCUMENTS_BATCH_SIZE": "129"}):
            r = update_documents(body=UpdateDocumentsBodyParams(documents=[{"_id": "1"}] * 129),
                                 index_name=self.structured_index_name, marqo_config=self.config)

    def test_proper_error_is_raised_for_unstructured_index(self):
        updated_doc = {
            "text_field_tensor": "I can't be updated",
            "_id": "1"
        }
        with self.assertRaises(UnsupportedFeatureError) as cm:
            r = update_documents(body=UpdateDocumentsBodyParams(documents=[updated_doc]),
                                 index_name=self.test_unstructured_index_name, marqo_config=self.config)

        self.assertIn("is not supported for unstructured indexes", str(cm.exception))