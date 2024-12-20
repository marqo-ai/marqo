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
        "indexName": "test_search_api_structured_index",
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
    }

    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index",
        "type": "unstructured",
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
    }

    docs = [
        {"_id": "1", "text_field": "a photo of a cat", "documentPopularity": 0.5 * 1 ** 39},
        {"_id": "2", "text_field": "a photo of a cat", "documentPopularity": 4.5 * 1 ** 39},
        {"_id": "3", "text_field": "a photo of a cat", "documentPopularity": 5.5 * 1 ** 39},
        {"_id": "4", "text_field": "a photo of a cat"}
    ]

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
        self.logger.info(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)
        errors = []  # Collect errors to report them at the end

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            tensor_fields = ["text_field"] if "unstr" in index.get("type") else None
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,                                                                               tensor_fields=tensor_fields)
            except Exception as e:
                errors.append((index, str(e)))

        all_results = {}
        # Loop through queries, search methods, and result keys to populate unstructured_results
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            all_results[index_name] = {}
            try:
                result = self.client.index(index_name).search(
                    q="",
                    score_modifiers={
                        "add_to_score": [{"field_name": "documentPopularity", "weight": 2}],
                    }
                )
                all_results[index_name] = result
            except Exception as e:
                errors.append((index_name, str(e)))

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
                        "add_to_score": [{"field_name": "documentPopularity", "weight": 2}],
                    }
                )
                self._compare_search_results(stored_results[index_name], result)
            except Exception as e:
                test_failures.append((index_name, str(e)))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in query {query}, search_method {search_method}, idx: {idx} : {error}"
                for query, search_method, idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")

    def _compare_search_results(self, expected_result, actual_result):
        """Compare two search results and assert if they match."""
        # We compare just the hits because the result contains other fields like processingTime which changes in every search API call.
        self.assertEqual(
            expected_result.get("hits")[0]["_score"],
            actual_result.get("hits")[0]["_score"],
            f'The _score field of the first hit for expected & actual result do not match. Expected: {expected_result.get("hits")[0]["_score"]}, Got: {actual_result.get("hits")[0]["_score"]}'
        )
        self.assertEqual(
            expected_result.get("hits")[0]["_id"],
            actual_result.get("hits")[0]["_id"],
            f'The _id field of the first hit for expected & actual result do not match. Expected: {expected_result.get("hits")[0]["_score"]}, Got: {actual_result.get("hits")[0]["_score"]}'
        )
