import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.15.0')
class TestSearchWithGlobalScoreModifiers(BaseCompatibilityTestCase):

    tensor_fields = ["text_field", "image_field"]
    structured_index_metadata =  {
        "indexName": "test_search_api_structured_index_global_score_modifiers",
        "type": "structured",
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
        "allFields": [
            {"name": "text_field", "type": "text", "features": ["lexical_search"]},
            {"name": "image_field", "type": "image_pointer"},
            {"name": "multiply_1", "type": "float", "features": ["score_modifier"]},
            {"name": "multiply_2", "type": "float", "features": ["score_modifier"]},
            {"name": "add_1", "type": "float", "features": ["score_modifier"]},
            {"name": "add_2", "type": "float", "features": ["score_modifier"]},
        ],
        "tensorFields": tensor_fields,
    }

    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index_global_score_modifiers",
        "type": "unstructured",
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
    }

    docs = [
        {"_id": "both1", "text_field": "dogs", "multiply_1": -1, "add_1": -1},  # HIGH tensor and lexical
        {"_id": "tensor1", "text_field": "puppies", "multiply_1": 2, "add_1": 2},  # MID tensor
        {"_id": "tensor2", "text_field": "random words", "multiply_1": 3, "add_1": 3},  # LOW tensor
    ]

    hybrid_test_cases = [
        ("disjunction", "rrf")
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
        self.logger.debug(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)
        errors = []  # Collect errors to report them at the end

        self.logger.debug(f'Feeding documents to {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            try:
                if index.get("type") is not None and index.get('type') == 'structured':
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs)
                else:
                    self.client.index(index_name=index['indexName']).add_documents(documents=self.docs,
                                                                                   tensor_fields=self.tensor_fields)
            except Exception as e:
                errors.append((index, traceback.format_exc()))

        all_results = {}
        # Loop through ranking/retrieval methods and result keys to populate results
        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            if index_name not in all_results:
                all_results[index_name] = {}

            for retrieval_method, ranking_method in self.hybrid_test_cases:
                try:
                    if retrieval_method not in all_results[index_name]:
                        all_results[index_name][retrieval_method] = {}
                        
                    result = self.client.index(index_name).search(
                        q="dogs",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": retrieval_method,
                            "rankingMethod": ranking_method
                        },
                        score_modifiers={
                            "multiply_score_by": [
                                {
                                    "field_name": "multiply_1",
                                    "weight": 2
                                }
                            ],
                            "add_to_score": [
                                {
                                    "field_name": "add_1",
                                    "weight": 3
                                }
                            ]
                        },
                        rerank_depth=2      # To show not all results are reranked
                    )
                    all_results[index_name][retrieval_method][ranking_method] = result
                    self.logger.debug(f"Result for {index_name} with retrieval method {retrieval_method} "
                                      f"and ranking method {ranking_method}: {result}")
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

    def test_search_with_global_score_modifiers(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            for retrieval_method, ranking_method in self.hybrid_test_cases:
                try:
                    result = self.client.index(index_name).search(
                        q="dogs",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": retrieval_method,
                            "rankingMethod": ranking_method
                        },
                        score_modifiers={
                            "multiply_score_by": [
                                {
                                    "field_name": "multiply_1",
                                    "weight": 2
                                }
                            ],
                            "add_to_score": [
                                {
                                    "field_name": "add_1",
                                    "weight": 3
                                }
                            ]
                        },
                        rerank_depth=2  # To show not all results are reranked
                    )
                    self.assertEqual(stored_results[index_name][retrieval_method][ranking_method].get("hits"),
                                     result.get("hits"))
                except Exception as e:
                    test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")