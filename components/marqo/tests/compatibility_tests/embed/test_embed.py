import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.5.0')
class TestEmbed(BaseCompatibilityTestCase):
    indexes_to_test_on = [{
        "indexName": "test_embed_api_index",
         "model": "hf/e5-base-v2"
    }]

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
        all_results = {}
        errors = []  # Collect any errors to report them at the end
        self.logger.debug(f'Embedding documents in {self.indexes_to_test_on}')
        for index in self.indexes_to_test_on:
            try:
                all_results[index['indexName']]  = self.client.index(index_name = index['indexName']).embed(
                content=[
                    "Men shoes brown",
                    {"Large grey hat": 0.7, "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg": 0.3}
                ],
                content_type=None
            )
            except Exception as e:
                errors.append((index['indexName'], traceback.format_exc()))
        if errors:
            formatted_errors = [f"Index: {index_name}, Error: {error}" for index_name, error in errors]
            self.logger.error("\n".join(formatted_errors))  # Fail the prepare method with all collected errors

        self.logger.debug(f"Ran prepare method for {self.indexes_to_test_on} inside test class {self.__class__.__name__}")
        self.save_results_to_file(all_results)

    def test_embed(self):
        self.logger.info(f"Running test_embed on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()
        test_failures = [] # this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
            try:
                expected_result = stored_results[index_name]
                actual_result = self.client.index(index_name).embed(                    content=[
                            "Men shoes brown",
                            {"Large grey hat": 0.7, "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg": 0.3}
                        ],
                        content_type=None
                )
                self.logger.debug(f"Printing expected result {expected_result}")
                self.logger.debug(f"Printing actual_result {actual_result}")
                self._compare_embed_results(expected_result, actual_result)

            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))
            if test_failures:
                failure_message = "\n".join([
                    f"Failure in index {idx}, {error}"
                    for idx, error in test_failures
                ])
                self.logger.error("\n".join(failure_message))  # Fail the prepare method with all collected errors





    def _compare_embed_results(self, expected_result, actual_result):
        self.assertEqual(expected_result.get("embeddings"), actual_result.get("embeddings"), f"Expected embeddings don't match actual embeddings. Expected {expected_result.get('embeddings')} but got {actual_result.get('embeddings')}")
        self.assertEqual(expected_result.get("content"), actual_result.get("content"), f"Expected results don't match actual results. Expected {expected_result.get('content')} but got {actual_result.get('content')}")