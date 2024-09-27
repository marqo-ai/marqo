import os
from unittest import mock

from marqo.api.models.update_documents import UpdateDocumentsBodyParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_document import UnstructuredVespaDocument
from marqo.tensor_search import tensor_search
from marqo.tensor_search.api import update_documents
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search.models.api_models import ScoreModifierLists
from tests.marqo_test import MarqoTestCase


class TestDictScoreModifiers(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='random/small'),
        )

        # STRUCTURED indexes
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name="random/small"),
            vector_numeric_type=VectorNumericType.Float,
            normalize_embeddings=True,
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="double_score_mods", type=FieldType.Double,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="float_score_mods", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods", type=FieldType.MapFloat,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_int", type=FieldType.MapInt,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_long", type=FieldType.MapLong,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_double", type=FieldType.MapDouble,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="score_mods_int", type=FieldType.Int,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="score_mods_long", type=FieldType.Long,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="price_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
            ],
            hnsw_config=HnswConfig(
                ef_construction=512,
                m=16
            ),
            tensor_fields=["text_field"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            structured_default_text_index
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.structured_default_text_index = cls.indexes[1]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    # Test Double score modifier
    def test_double_score_modifier(self):
        """
        Test that adding to score works for a double score modifier.
        """
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "double_score_mods": 0.5 * 1 ** 39},
                            {"_id": "2", "text_field": "a photo of a cat", "double_score_mods": 4.5 * 1 ** 39},
                            {"_id": "3", "text_field": "a photo of a cat", "double_score_mods": 5.5 * 1 ** 39},
                            {"_id": "4", "text_field": "a photo of a cat"}
                        ]
                    )
                )
                # Search with score modifier
                # 0.5 + 5.5 * 2 = 11.5
                score_modifier = ScoreModifierLists(**{"add_to_score": [{"field_name": "double_score_mods", "weight": 2}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )
                # Get the score of the first result and divide by 1**39
                score_of_first_result = res["hits"][0]["_score"] / 1 ** 39
                # Assert that the first result has _id "3" and 11 <= score <= 12
                self.assertEqual(res["hits"][0]["_id"], "3")
                self.assertTrue(11 <= score_of_first_result <= 12)

    # Test Long score modifier
    def test_long_score_modifier(self):
        """
        Test that adding to score works for a long score modifier.
        """
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "score_mods_long": 2 ** 34},
                            {"_id": "2", "text_field": "a photo of a cat", "score_mods_long": 2 ** 35},
                            {"_id": "3", "text_field": "a photo of a cat", "score_mods_long": 2 ** 36},
                            {"_id": "4", "text_field": "a photo of a cat"}
                        ]
                    )
                )
                # Search with score modifier
                # 0.5 + 2**36 * 2 = 2**37
                score_modifier = ScoreModifierLists(**{"add_to_score": [{"field_name": "score_mods_long", "weight": 2}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )
                # Get the score of the first result and divide by 1**39
                score_of_first_result = res["hits"][0]["_score"] / 1 ** 39
                # Assert that the first result has _id "3" and 2**37-1 <= score <= 2**37+1
                self.assertEqual(res["hits"][0]["_id"], "3")
                self.assertTrue(2 ** 37 - 1 <= score_of_first_result <= 2 ** 37 + 1)

    # Test Add to score
    def test_add_to_score_map_score_modifier(self):
        """
        Test that adding to score works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                )
                # Search with score modifier
                # 0.5 + 1 * 5 = 5.5
                score_modifier = ScoreModifierLists(
                    **{"add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 5}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 5 <= score <= 6
                self.assertIn(res["hits"][0]["_id"], ["6", "7"])
                self.assertTrue(5 <= score_of_first_result <= 6)

    # Test multiply score by
    def test_multiply_score_by_map_score_modifier(self):
        """
        Test that multiplying score by works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7)
                score_modifier = ScoreModifierLists(
                    **{"multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4},
                                             {"field_name": "map_score_mods.d", "weight": 4}]}) # Nonexistent field.
                                            # Nonexistent field should not zero out the whole score
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 0.8 <= score <= 1.2
                self.assertIn(res["hits"][0]["_id"], ["1", "7"])
                self.assertTrue(0.8 <= score_of_first_result <= 1.2)

    # Test combined add to score and multiply score by
    def test_combined_map_score_modifier(self):
        """
        Test that combining adding to score and multiplying score by works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7) multiply_score_by
                # for 7, 0.5 * 0.5 * 4 + 1 * 2 = 3 (6 and 7) add_to_score
                score_modifier = ScoreModifierLists(**{
                    "add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 2}],
                    "multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4}]
                }
                                                    )
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 5 <= score <= 6
                self.assertTrue(res["hits"][0]["_id"] == "7")
                self.assertTrue(2.9 <= score_of_first_result <= 3.1)

    def test_partial_document_update(self):
        """
        Test that partial document update works for a map score modifier.
        """
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                )

                # Get Document and assert that the score modifier is 0.5
                # Sample:  updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
                original_doc = tensor_search.get_document_by_id(self.config, index.name, "1")
                self.assertTrue(original_doc["_id"] == "1")
                self.assertTrue(original_doc["map_score_mods"]["a"] == 0.5)

                # Update the document
                updated_doc = {
                    "map_score_mods": {"a": 1.5},
                    "_id": "1"
                }
                r = update_documents(
                    marqo_config=self.config,
                    index_name=index.name,
                    body=UpdateDocumentsBodyParams(documents=[updated_doc])
                )

                # Get updated document and assert that the score modifier is 1.5
                updated_doc = tensor_search.get_document_by_id(self.config, index.name, "1")
                self.assertTrue(updated_doc["_id"] == "1")
                self.assertTrue(updated_doc["map_score_mods"]["a"] == 1.5)

                # Search with score modifier
                # 0.5 + 1.5 * 2 = 3.5
                score_modifier = ScoreModifierLists(**{"add_to_score": [{"field_name": "map_score_mods.a", "weight": 2}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )
                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "1" and 3 <= score <= 4
                self.assertTrue(res["hits"][0]["_id"] == "1")
                self.assertTrue(3 <= score_of_first_result <= 4)

    def test_long_dict_score_modifier(self):
        """
        Test that long score modifier works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods_long": {"a": 4294967295012}},
                            {"_id": "2", "text_field": "a photo of a cat", "score_mods_long": 4294967295012},
                            {"_id": "4", "text_field": "a photo of a cat", "score_mods_long": 1},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7)
                score_modifier = ScoreModifierLists(**{
                    "add_to_score": [{"field_name": "map_score_mods_long.a", "weight": 20},
                                     {"field_name": "score_mods_long", "weight": 20}],
                }
                                                    )
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]
                score_of_second_result = res["hits"][1]["_score"]

                # anticipated score is 4294967295012*20 = 85899345900240
                # Assert that the first and second result both have _ids in 1 and 2
                # and 85899345900239 <= score <= 85899345900241
                self.assertTrue(res["hits"][0]["_id"] in ["2", "1"])
                self.assertTrue(res["hits"][1]["_id"] in ["2", "1"])
                self.assertTrue(85899345900239 <= score_of_first_result <= 85899345900241)
                self.assertTrue(85899345900239 <= score_of_second_result <= 85899345900241)

    def test_unstructured_unsupported_map_error(self):
        """
        Test that only the document errors out, not the whole batch
        """
        for index in [self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods_long": {"a": 4294967295012}},
                            {"_id": "2", "text_field": "a photo of a cat", "score_mods_long": 4294967295012},
                            {"_id": "3", "text_field": "a photo of a cat", "score_mods_long": 1,
                             "unsupported_map": {"text": "hello", "number": 42}},
                            {"_id": "4", "text_field": "a photo of a cat", "score_mods_long": 1,
                             "supported_number_map": {"number": 42}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods_float": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                    )
                ).dict(exclude_none=True, by_alias=True)

                # Expected
                expected = {"1": 200, "2": 200, "3": 400, "4": 200, "5": 200}
                expected_doc_3_error = {
                    "invalid_argument": "Value 'hello' for key 'text' in map field is not of type int or float."}

                # Get actual returned document add status
                actual = {item["_id"]: item["status"] for item in res["items"]}
                actual_doc3 = {res["items"][2]["code"]: res["items"][2]["error"]}

                # Assert that expected and actual status codes are equal
                self.assertEqual(expected, actual)
                self.assertIn(expected_doc_3_error["invalid_argument"], actual_doc3["invalid_argument"])

    def test_unstructured_wrong_map_numerical_format(self):
        """
        Test that the wrong map numerical format errors out
        """
        for index in [self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods_long": {"a": [42]}},
                            {"_id": "2", "text_field": "a photo of a cat", "score_mods_long": {1: 1}},
                            {"_id": "3", "text_field": "a photo of a cat", "score_mods_long": 84},
                            {"_id": "4", "text_field": "a photo of a cat", "score_mods_long": 1,
                             "unsupported_map": {"text": "hello", "number": 42}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                             "map_score_mods_float": {"a": 0.5}},
                            {"_id": "6", "text_field": "a photo of a dog", "my_int": 2,
                             "my_custom_audio_vector_1": {"vector": [x for x in range(32)]}},
                        ],
                        tensor_fields=["text_field", "my_combination_field", "my_custom_audio_vector_1"],
                        mappings={
                            "my_combination_field": {
                                "type": "multimodal_combination",
                                "weights": {"text2": 0.5, "text_field": 0.5},
                            },
                            "my_custom_audio_vector_1": {"type": "custom_vector"},
                        },
                    )
                ).dict(exclude_none=True, by_alias=True)

                # Expected
                expected_results = {
                    "1": {"status": 400, "error": "in map field is not of type int or float",
                          "code": "invalid_argument"},
                    "2": {"status": 400, "error": "All keys must be strings.", "code": "invalid_argument"},
                    "3": {"status": 200},
                    "4": {"status": 400, "error": "in map field is not of type int or float",
                          "code": "invalid_argument"},
                    "5": {"status": 200},
                    "6": {"status": 200}
                }

                # Get actual returned document
                actual_results = res["items"]

                for item in actual_results:
                    doc_id = item["_id"]
                    expected = expected_results[doc_id]

                    with self.subTest(doc_id=doc_id):
                        self.assertEqual(item["status"], expected["status"],
                                         f"Document {doc_id} status mismatch")

                        if expected["status"] == 400:
                            self.assertIn(expected["error"], item["error"],
                                          f"Document {doc_id} error message mismatch")
                        else:
                            self.assertIn("_id", item, f"Document {doc_id} missing _id field")

                actual_statuses = {item["_id"]: item["status"] for item in res["items"]}
                expected_statuses = {id: data["status"] for id, data in expected_results.items()}
                self.assertEqual(expected_statuses, actual_statuses, "Overall status results mismatch")

    def test_unstructured_map_numerical_as_custom_vector(self):
        """
        Test that the wrong map numerical format errors out
        """
        for index in [self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                with self.assertRaises(ValidationError):
                    res = self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index.name,
                            docs=[
                                {"_id": "7", "text2": "hello", "my_int": 2, "my_custom_audio_vector_1": {"vector": 24}},
                            ],
                            tensor_fields=["text_field", "my_custom_audio_vector_1"],
                            mappings={
                                "my_custom_audio_vector_1": {"type": "custom_vector"},
                            },
                        )
                    )

    def test_structured_tensor_storage_old_and_new(self):
        # Mock the index and field map
        index = mock.MagicMock()
        index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.9.0")
        index.field_map = {
            "float_field": Field(name="float_field", type=FieldType.Float, features=[FieldFeature.ScoreModifier]),
            "map_float_field": Field(name="map_float_field", type=FieldType.MapFloat,
                                     features=[FieldFeature.ScoreModifier]),
            "double_field": Field(name="double_field", type=FieldType.Double, features=[FieldFeature.ScoreModifier]),
            "map_double_field": Field(name="map_double_field", type=FieldType.MapDouble,
                                      features=[FieldFeature.ScoreModifier]),
        }

        vespa_index = StructuredVespaIndex(marqo_index=index)

        # Document to be processed
        marqo_document = {
            "float_field": 1.23,
            "double_field": 4.56,
            "map_float_field": {"float_field": 1.23},
            "map_double_field": {"double_field": 4.56},
        }

        # Call the method under test
        vespa_doc = vespa_index.to_vespa_document(marqo_document)

        # Assertions to check if the fields are stored correctly
        self.assertIn("marqo__score_modifiers_float", vespa_doc["fields"])
        self.assertIn("marqo__score_modifiers_double_long", vespa_doc["fields"])
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers_float"]["float_field"], 1.23)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers_float"]["map_float_field.float_field"], 1.23)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers_double_long"]["double_field"], 4.56)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers_double_long"]["map_double_field.double_field"],
                         4.56)

        # Test with a version less than 2.9.0
        index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.8.0")
        vespa_index = StructuredVespaIndex(marqo_index=index)
        vespa_doc = vespa_index.to_vespa_document(marqo_document)

        # Assertions for version less than 2.9.0
        self.assertIn("marqo__score_modifiers", vespa_doc["fields"])
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers"]["float_field"], 1.23)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers"]["double_field"], 4.56)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers"]["map_float_field.float_field"], 1.23)
        self.assertEqual(vespa_doc["fields"]["marqo__score_modifiers"]["map_double_field.double_field"], 4.56)

    def test_unstructured_from_marqo_document(self):
        # Mock document
        marqo_document = {
            "_id": "1",
            "MARQO_DOC_ID": "doc1",
            "float_field": 1.23,
            "double_field": 4.56e39,  # This value is intentionally large to simulate a double,
            "map_double_field": {"double_field": 4.56e39},
            "map_float_field": {"float_field": 1.23},
            "int_field": 42
        }

        # Test with Marqo version >= 2.9.0
        marqo_index_version = semver.VersionInfo.parse("2.9.0")
        document = UnstructuredVespaDocument.from_marqo_document(
            marqo_document, filter_string_max_length=100
        )

        # Assertions for version >= 2.9.0
        self.assertEqual(document.fields.score_modifiers_fields["float_field"], 1.23)
        self.assertEqual(document.fields.score_modifiers_fields["double_field"], 4.56e39)
        self.assertEqual(document.fields.score_modifiers_fields["map_double_field.double_field"], 4.56e39)
        self.assertEqual(document.fields.score_modifiers_fields["map_float_field.float_field"], 1.23)
        self.assertIn("int_field", document.fields.score_modifiers_fields)
