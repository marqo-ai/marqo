import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.9.0')
class TestSearchWithScoreModifiers(BaseCompatibilityTestCase):

    image_model = 'open_clip/ViT-B-32/laion2b_s34b_b79k'
    multimodal_weights = {"image_field": 0.9, "text_field": 0.1}
    mappings = {
        "multimodal_field": {
            "type": "multimodal_combination",
            "weights": multimodal_weights,
        }
    }
    tensor_fields = ["multimodal_field", "text_field", "image_field"]
    structured_index_metadata =  {
        "indexName": "test_search_api_structured_index_score_modifiers",
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
            {"name": "text", "type": "text", "features":["lexical_search"]},
            {"name": "rating", "type": "double", "features": ["score_modifier"]},
            {"name": "popularity", "type": "int", "features": ["score_modifier"]},
            {"name":"category", "type":"text", "features":["filter"]}
            # test no whitespace
        ],
        "tensorFields": ["text_field"],
        "annParameters": {
            "spaceType": "prenormalized-angular",
            "parameters": {"efConstruction": 512, "m": 16},
        }
    }

    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index_score_modifiers",
        "type": "unstructured",
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
    }

    docs_with_double_and_long_score_modifiers = [
        {"_id": "1", "text_field": "a photo of a cat", "double_score_mods": 0.5 * 1 ** 39, "long_score_mods": 3 * 1 ** 39},
        {"_id": "2", "text_field": "a photo of a cat", "double_score_mods": 4.5 * 1 ** 39, "long_score_mods": 4 * 1 ** 39},
        {"_id": "3", "text_field": "a photo of a cat", "double_score_mods": 5.5 * 1 ** 39, "long_score_mods": 5 * 1 ** 39},
        {"_id": "4", "text_field": "a photo of a cat"}
    ]

    docs_with_float_and_int_score_modifiers = [
        {
            "_id": "doc1",
            "text_field": "A great product with many features",
            "rating": 4.5,
            "popularity": 120,
            "category": "electronics",
        },
        {
            "_id": "doc2",
            "text_field": "Another decent product but slightly less popular",
            "rating": 3.8,
            "popularity": 80,
            "category": "electronics"
        },
        {
            "_id": "doc3",
            "text_field": "An older, but very popular product",
            "rating": 4.0,
            "popularity": 150,
            "category": "home"
        },
        {
            "_id": "doc4",
            "text_field": "A basic product not popular",
            "rating": 2.0,
            "popularity": 20,
            "category": "home"
        }
    ]

    docs = docs_with_double_and_long_score_modifiers + docs_with_float_and_int_score_modifiers

    indexes_to_test_on = [structured_index_metadata, unstructured_index_metadata]

    # We need to set indexes_to_delete variable in an overriden tearDownClass() method
    # So that when the test method has finished running, pytest is able to delete the indexes added in
    # prepare method of this class
    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [index['indexName'] for index in cls.indexes_to_test_on]
        super().setUpClass()

    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        self.logger.debug(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)
        errors = []  # Collect errors to report them at the end

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            tensor_fields = ["text_field"] if "unstr" in index.get("type") else None
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,
                                                                                   tensor_fields=tensor_fields)
            except Exception as e:
                errors.append((index, traceback.format_exc()))

        all_results = {}
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for type_of_score_modifier in ["double_score_mods", "long_score_mods", "rating", "popularity"]:
            for index in self.indexes_to_test_on:
                index_name = index['indexName']
                if index_name not in all_results:
                    all_results[index_name] = {}

                try:
                    result = self.client.index(index_name).search(
                        q="",
                        score_modifiers={"add_to_score": [
                            {
                                "field_name": type_of_score_modifier,
                                "weight": 2
                            }
                        ]
                        }
                    )
                    all_results[index_name][type_of_score_modifier] = result
                except Exception as e:
                    errors.append((index_name, traceback.format_exc()))

        if errors:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")

        self.save_results_to_file(all_results)
        # store the result of search across all structured & unstructured indexes

    def test_search_with_double_score_modifier(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                result = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                    "add_to_score": [{"field_name": "double_score_mods", "weight": 2}],
                    }
                )
                self.assertEqual(stored_results[index_name]["double_score_mods"].get("hits"), result.get("hits"))
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in query idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def test_search_with_long_score_modifier(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                result = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "long_score_mods", "weight": 2}],
                    }
                )
                self.assertEqual(stored_results[index_name]["long_score_mods"].get("hits"), result.get("hits"))
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def test_search_with_int_score_modifier(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                result = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "rating", "weight": 2}],
                    }
                )
                self.assertEqual(stored_results[index_name]["rating"].get("hits"), result.get("hits"))
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def test_search_with_float_score_modifier(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                result = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "popularity", "weight": 2}],
                    }
                )
                self.assertEqual(stored_results[index_name]["popularity"].get("hits"), result.get("hits"))
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")