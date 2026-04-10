import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase


@pytest.mark.marqo_version('2.26.0')
class TestSearchWithCustomScoreRerankers(BaseCompatibilityTestCase):
    """
    We only test unstructured and hybrid RRF.
    """

    tensor_fields = ["tensor_retrieval_field", "tensor_ranking_field"]
    unstructured_index_metadata = {
        "indexName": "test_search_api_unstructured_index_custom_score_rerankers",
        "type": "unstructured",
        "model": "open_clip/ViT-B-16-SigLIP/webli"
    }

    docs = [
        {
            # (1) In BOTH tensor and lexical
            "_id": "doc1",
            "lex_retrieval_field": "tuxedo tuxedo tuxedo",  # VERY HIGH lexical score
            "tensor_retrieval_field": "tuxedo",             # VERY HIGH tensor score
            "lex_ranking_field": "tuxedo",                  # lowest bm25 score for global reranking
            "tensor_ranking_field": "unrelated",            # lowest closeness score for global reranking
        },
        {
            # (2) In ONLY tensor (medium strength)
            "_id": "doc2",
            "lex_retrieval_field": "no match",              # no lexical match
            "tensor_retrieval_field": "suit",                # MEDIUM tensor score
            "lex_ranking_field": "tuxedo tuxedo",           # 2nd lowest bm25 score for global reranking
            "tensor_ranking_field": "rainbow tie",          # 2nd lowest closeness score for global reranking
        },
        {
            # (3) In ONLY lexical (medium strength)
            "_id": "doc3",
            "lex_retrieval_field": "tuxedo tuxedo",         # MEDIUM lexical score
            "tensor_retrieval_field": "unrelated",          # no tensor match for retrieval
            "lex_ranking_field": "tuxedo tuxedo tuxedo",    # 3rd lowest bm25 score for global reranking
            "tensor_ranking_field": "shorts",               # 3rd lowest closeness score for global reranking
        },
        {
            # (4) In ONLY tensor (lower strength)
            "_id": "doc4",
            "lex_retrieval_field": "no match",              # no lexical match
            "tensor_retrieval_field": "shorts",             # LOWER tensor score (but it's still clothes)
            "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo",  # 4th lowest bm25 score for global reranking
            "tensor_ranking_field": "suit",                      # 4th lowest closeness score for global reranking
        },
        {
            # (5) In ONLY lexical (lower strength)
            "_id": "doc5",
            "lex_retrieval_field": "tuxedo",                # LOW lexical score
            "tensor_retrieval_field": "backpack",           # no tensor match for retrieval
            "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo",  # highest bm25 score for global reranking
            "tensor_ranking_field": "tuxedo",               # highest closeness score for global reranking
        },
    ]

    hybrid_test_cases = [
        ("disjunction", "rrf")
    ]

    indexes_to_test_on = [unstructured_index_metadata]

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
                        q="tuxedo",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": retrieval_method,
                            "rankingMethod": ranking_method
                        },
                        score_modifiers={
                            "multiply_score_by": [
                                {
                                    "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                                    "weight": 1.5,
                                }
                            ],
                            "add_to_score": [
                                {
                                    "field_name": "marqo__score_bm25_sum",
                                    "weight": 2.5
                                }
                            ]
                        },
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

    def test_search_with_custom_score_rerankers(self):
        """Run search queries and compare the results with the stored results."""
        self.logger.info(f"Running test_search on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] #this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions.

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            for retrieval_method, ranking_method in self.hybrid_test_cases:
                try:
                    result = self.client.index(index_name).search(
                        q="tuxedo",
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": retrieval_method,
                            "rankingMethod": ranking_method
                        },
                        score_modifiers={
                            "multiply_score_by": [
                                {
                                    "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                                    "weight": 1.5,
                                }
                            ],
                            "add_to_score": [
                                {
                                    "field_name": "marqo__score_bm25_sum",
                                    "weight": 2.5
                                }
                            ]
                        },
                    )
                    self._compare_search_results(stored_results[index_name][retrieval_method][ranking_method], result)
                except Exception as e:
                    test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in idx: {idx} : {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")