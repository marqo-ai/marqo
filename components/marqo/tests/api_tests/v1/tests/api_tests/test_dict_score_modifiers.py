import uuid
import math

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError


class TestDictScoreModifiers(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "vectorNumericType": "float",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": "sentence",
                },
                "imagePreprocessing": {"patchMethod": None},
                "allFields": [
                    {"name": "text_field", "type": "text", "features": ["lexical_search"]},
                    {"name": "double_score_mods", "type": "double", "features": ["score_modifier"]},
                    {"name": "long_score_mods", "type": "long", "features": ["score_modifier"]},
                    {"name": "map_score_mods", "type": "map<text, float>", "features": ["score_modifier"]},
                    {"name": "map_score_mods_int", "type": "map<text,int>", "features": ["score_modifier"]},
                    # test no whitespace
                ],
                "tensorFields": ["text_field"],
                "annParameters": {
                    "spaceType": "prenormalized-angular",
                    "parameters": {"efConstruction": 512, "m": 16},
                }
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    # Test Double score modifier
    def test_double_score_modifier(self):
        """
        Test that adding to score works for a double score modifier.
        """
        test_cases = [self.structured_index_name, self.unstructured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "double_score_mods": 0.5 * 1 ** 39},
                    {"_id": "2", "text_field": "a photo of a cat", "double_score_mods": 4.5 * 1 ** 39},
                    {"_id": "3", "text_field": "a photo of a cat", "double_score_mods": 5.5 * 1 ** 39},
                    {"_id": "4", "text_field": "a photo of a cat"}
                ]
                tensor_fields = ["text_field"] if "unstr" in test_index_name else None
                res = self.client.index(test_index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                # Search
                # 0.5 + 5.5 * 2 = 11.5
                res = self.client.index(test_index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "double_score_mods", "weight": 2}]
                    }
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
        test_cases = [self.structured_index_name, self.unstructured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "long_score_mods": 2 ** 34},
                    {"_id": "2", "text_field": "a photo of a cat", "long_score_mods": 2 ** 35},
                    {"_id": "3", "text_field": "a photo of a cat", "long_score_mods": 2 ** 36},
                    {"_id": "4", "text_field": "a photo of a cat"}
                ]
                tensor_fields = ["text_field"] if "unstr" in test_index_name else None
                res = self.client.index(test_index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                # Search
                # 0.5 + 2**36 * 2 = 2**37
                res = self.client.index(test_index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "long_score_mods", "weight": 2}]
                    }
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
        test_cases = [self.structured_index_name, self.unstructured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                    {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                    {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                    {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                    {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                    {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                    {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                     "map_score_mods": {"a": 0.5}},
                ]

                tensor_fields = ["text_field"] if "unstr" in test_index_name else None
                res = self.client.index(test_index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                # Search
                # 0.68 + 1 * 5 = 5.68
                res = self.client.index(test_index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 5}],
                    }
                )

                # Assert that the first result is either 6 or 7
                first_result_id = res["hits"][0]["_id"]
                self.assertTrue(first_result_id in ["6", "7"])

                # Assert that 5 <= _score <= 6
                first_result_score = res["hits"][0]["_score"]
                self.assertTrue(5 <= first_result_score <= 6)

    # Test multiply score by
    def test_multiply_score_by_map_score_modifier(self):
        """
        Test that multiplying score by works for a map score modifier.
        """
        test_cases = [self.structured_index_name, self.unstructured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                    {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                    {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                    {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                    {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                    {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                    {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                     "map_score_mods": {"a": 0.5}},
                ]

                tensor_fields = ["text_field"] if "unstr" in test_index_name else None
                res = self.client.index(test_index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                # Search
                # 0.68 * 0.5 * 4 = 1.36 (1 and 7)
                res = self.client.index(test_index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4}]
                    }
                )

                # Assert that the first result is either 1 or 7
                first_result_id = res["hits"][0]["_id"]
                self.assertTrue(first_result_id in ["1", "7"])

                # Assert that 1 <= _score <= 1.5
                first_result_score = res["hits"][0]["_score"]
                self.assertTrue(1 <= first_result_score <= 1.5)

    # Test combined add to score and multiply score by
    def test_combined_map_score_modifier(self):
        """
        Test that combining adding to score and multiplying score by works for a map score modifier.
        """
        test_cases = [self.structured_index_name, self.unstructured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                    {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                    {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                    {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                    {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                    {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                    {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                     "map_score_mods": {"a": 0.5}},
                ]

                tensor_fields = ["text_field"] if "unstr" in test_index_name else None
                res = self.client.index(test_index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                # Search
                # 0.68 * 1 * 4 = 2.72
                # 0.68 * 0.5 * 4 + 1 * 2 = 3.36
                res = self.client.index(test_index_name).search(
                    q="photo",
                    score_modifiers={
                        "add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 2}],
                        "multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4}]
                    }
                )

                # Assert that the first result is  7
                first_result_id = res["hits"][0]["_id"]
                self.assertTrue(first_result_id in ["7"])

                # Assert that 3 <= _score <= 3.5
                first_result_score = res["hits"][0]["_score"]
                self.assertTrue(3 <= first_result_score <= 3.5)

    def test_partial_document_update(self):
        """
        Test that partial document update works for a map score modifier.
        """
        test_cases = [self.structured_index_name]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                docs = [
                    {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                    {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                    {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                    {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                    {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                    {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                    {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                     "map_score_mods": {"a": 0.5}},
                ]

                # Add documents
                res = self.client.index(test_index_name).add_documents(documents=docs)

                # Get document and assert that the score modifier is 0.5
                res = self.client.index(test_index_name).get_documents(
                    document_ids=["1"]
                )
                self.assertTrue(res["results"][0]["_id"] == "1")
                self.assertTrue(res["results"][0]["map_score_mods"]["a"] == 0.5)

                # Update the document
                res = self.client.index(test_index_name).update_documents(
                    documents=[{"_id": "1", "map_score_mods": {"a": 1.5}}]
                )

                # Fetch Document
                res = self.client.index(test_index_name).get_documents(
                    document_ids=["1"]
                )

                # Assert that the document has been updated
                self.assertTrue(res["results"][0]["_id"] == "1")
                self.assertTrue(res["results"][0]["map_score_mods"]["a"] == 1.5)

                # Search with score modifiers
                # 0.68 + 1.5 * 2 = 3.88
                res = self.client.index(test_index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "map_score_mods.a", "weight": 2}],
                    }
                )
                # Assert that the first result is  1
                first_result_id = res["hits"][0]["_id"]
                self.assertTrue(first_result_id in ["1"])

                # Assert that 3 <= _score <= 4
                first_result_score = res["hits"][0]["_score"]
                self.assertTrue(3 <= first_result_score <= 4)

    def test_multiple_map_values_score_modifiers(self):
        """Test multiple map values score modifiers"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "map_score_mods": {"a": 1.5, "b": 2, "c":5}},
                    {"_id": "2", "text_field": "test", "map_score_mods": {"a": 1.5, "b": 2}},
                    {"_id": "3", "text_field": "test", "map_score_mods": {"a": 1.5}},
                    {"_id": "4", "text_field": "test"},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [
                            {"field_name": "map_score_mods.a", "weight": 2},
                        ],
                        "add_to_score": [
                            {"field_name": "map_score_mods.b", "weight": 1},
                            {"field_name": "map_score_mods.c", "weight": 3},
                        ]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score * (1.5 * 2) + (2 * 1) + (5 * 3) = base_score * 3 + 17
                # Doc 2: base_score * (1.5 * 2) + (2 * 1) = base_score * 3 + 2
                # Doc 3: base_score * (1.5 * 2) = base_score * 3
                # Doc 4: base_score
                base_score = 0.845687427
                expected_score_doc1 = base_score * 3 + 17
                expected_score_doc2 = base_score * 3 + 2
                expected_score_doc3 = base_score * 3

                tolerance = 0.1
                self.assertTrue(abs(res["hits"][0]["_score"] - expected_score_doc1) <= tolerance)
                self.assertTrue(abs(res["hits"][1]["_score"] - expected_score_doc2) <= tolerance)
                self.assertTrue(abs(res["hits"][2]["_score"] - expected_score_doc3) <= tolerance)

    def test_combination_score_modifiers(self):
        """Test combination of standard score modifier fields and maps"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 2.0, "map_score_mods": {"a": 1.5}},
                    {"_id": "2", "text_field": "test", "long_score_mods": 3, "map_score_mods_int": {"b": 2}},
                    {"_id": "3", "text_field": "test"},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [
                            {"field_name": "double_score_mods", "weight": 1},
                            {"field_name": "map_score_mods.a", "weight": 2},
                        ],
                        "add_to_score": [
                            {"field_name": "long_score_mods", "weight": 1},
                            {"field_name": "map_score_mods_int.b", "weight": 1},
                        ]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score * 2.0 * (1.5 * 2) = base_score * 6 
                # Doc 2: base_score + (3 * 1) + (2 * 1) = base_score + 5
                base_score = 0.845687427
                expected_score_doc1 = base_score * 6
                expected_score_doc2 = base_score + 5

                tolerance = 0.1
                self.assertTrue(abs(res["hits"][0]["_score"] - expected_score_doc2) <= tolerance)
                self.assertTrue(abs(res["hits"][1]["_score"] - expected_score_doc1) <= tolerance)

    def test_missing_score_modifiers(self):
        """Test cases where some or all specified score modifiers are missing in document"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 2.0},
                    {"_id": "2", "text_field": "test", "map_score_mods": {"a": 1.5}},
                    {"_id": "3", "text_field": "test"},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [
                            {"field_name": "double_score_mods", "weight": 2},
                            {"field_name": "map_score_mods.a", "weight": 2},
                            {"field_name": "map_score_mods.b", "weight": 2},
                        ]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score * (2.0 * 2) = base_score * 4
                # Doc 2: base_score * (1.5 * 2) = base_score * 3
                # Doc 3: base_score (no matching fields)
                base_score = 0.845687427
                expected_score_doc1 = base_score * 4
                expected_score_doc2 = base_score * 3

                self.assertAlmostEqual(res["hits"][0]["_score"], expected_score_doc1, places=2)
                self.assertAlmostEqual(res["hits"][1]["_score"], expected_score_doc2, places=2)
                self.assertNotEqual(res["hits"][2]["_score"], 0)  # Make sure that doc 3 has a non-zero score

    def test_empty_map_score_modifier(self):
        """Test behavior when a map score modifier is empty"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "map_score_mods": {}},
                    {"_id": "2", "text_field": "test", "map_score_mods": {"a": 1.5}},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 2}]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score (empty map, no modification)
                # Doc 2: base_score * (1.5 * 2) = base_score * 3
                base_score = 0.845687427
                expected_score_doc2 = base_score * 3

                self.assertEqual(res["hits"][0]["_id"], "2")
                self.assertAlmostEqual(res["hits"][0]["_score"], expected_score_doc2, places=2)
                self.assertGreater(res["hits"][0]["_score"], res["hits"][1]["_score"])

    def test_nonexistent_score_modifier(self):
        """Test behavior when a specified score modifier doesn't exist in the document"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 2.0},
                    {"_id": "2", "text_field": "test", "map_score_mods": {"a": 1.5}},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)
                
                try:
                    res = self.client.index(index_name).search(
                        q="",
                        score_modifiers={
                            "multiply_score_by": [{"field_name": "nonexistent_field", "weight": 2}]
                        }
                    )
                except MarqoWebError as e:
                    print(f"Caught MarqoWebError: {e}")
                    self.assertIn('has no score modifier field nonexistent_field', str(e))
                except Exception as e:
                    print(f"Unexpected exception: {e}")
                    raise

    def test_score_modifier_with_zero_value(self):
        """Test behavior when a score modifier has a zero value"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 0.0},
                    {"_id": "2", "text_field": "test", "double_score_mods": 1.0},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [{"field_name": "double_score_mods", "weight": 1}]
                    }
                )
                
                # Expected score calculation:
                # Doc 1: base_score * 0.0 = 0
                # Doc 2: base_score * 1.0 = base_score
                base_score = 0.845687427
                self.assertGreater(res["hits"][0]["_score"], res["hits"][1]["_score"])
                self.assertNotEqual(res["hits"][0]["_score"], 0)
                self.assertLess(res["hits"][1]["_score"], base_score)

    def test_score_modifier_with_negative_value(self):
        """Test behavior when a score modifier has a negative value"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": -1.0},
                    {"_id": "2", "text_field": "test", "double_score_mods": 1.0},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "double_score_mods", "weight": 1}]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score + (-1.0 * 1) = base_score - 1
                # Doc 2: base_score + (1.0 * 1) = base_score + 1
                base_score = 0.845687427
                expected_score_doc1 = base_score - 1
                expected_score_doc2 = base_score + 1

                self.assertAlmostEqual(res["hits"][0]["_score"], expected_score_doc2, places=2)
                self.assertAlmostEqual(res["hits"][1]["_score"], expected_score_doc1, places=2)
                self.assertGreater(res["hits"][0]["_score"], res["hits"][1]["_score"])

    def test_score_modifier_with_large_values(self):
        """Test behavior with very large score modifier values"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 1e20},
                    {"_id": "2", "text_field": "test", "double_score_mods": 1e10},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [{"field_name": "double_score_mods", "weight": 1}]
                    }
                )

                # Expected score calculation:
                # Doc 1: base_score * 1e20
                # Doc 2: base_score * 1e10
                # The exact values might cause overflow, so we'll check the ratio
                score_ratio = res["hits"][0]["_score"] / res["hits"][1]["_score"]
                expected_ratio = 1e10  # (1e20 / 1e10)

                self.assertGreater(res["hits"][0]["_score"], res["hits"][1]["_score"])
                self.assertAlmostEqual(math.log10(score_ratio), math.log10(expected_ratio), places=1)

    def test_multiple_score_modifiers_same_field(self):
        """Test using multiple score modifiers on the same field"""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                docs = [
                    {"_id": "1", "text_field": "test", "double_score_mods": 2.0},
                    {"_id": "2", "text_field": "test", "double_score_mods": 3.0},
                ]
                tensor_fields = ["text_field"] if "unstr" in index_name else None
                self.client.index(index_name).add_documents(documents=docs, tensor_fields=tensor_fields)

                res = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "multiply_score_by": [{"field_name": "double_score_mods", "weight": 2}],
                        "add_to_score": [{"field_name": "double_score_mods", "weight": 1}]
                    }
                )
                
                # Expected score calculation:
                # Doc 1: (base_score * 2.0 * 2) + (2.0 * 1) = base_score * 4 + 2
                # Doc 2: (base_score * 3.0 * 2) + (3.0 * 1) = base_score * 6 + 3
                base_score = 0.845687427
                expected_score_doc1 = base_score * 4 + 2
                expected_score_doc2 = base_score * 6 + 3

                self.assertAlmostEqual(res["hits"][0]["_score"], expected_score_doc2, places=2)
                self.assertAlmostEqual(res["hits"][1]["_score"], expected_score_doc1, places=2)
                self.assertGreater(res["hits"][0]["_score"], res["hits"][1]["_score"])
