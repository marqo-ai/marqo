import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
class TestEmbed(BaseCompatibilityTestCase):
    indexes_to_test_on = [{
        "indexName": "test_embed_api_index",
         "model": "hf/e5-base-v2"
    }]
    def prepare(self):
        """
        Prepare the indexes and add documents for the test.
        Also store the search results for later comparison.
        """
        self.logger.info(f"Creating indexes {self.indexes_to_test_on}")
        self.create_indexes(self.indexes_to_test_on)
        all_results = {}
        try:
            self.logger.debug(f'Embedding documents in {self.indexes_to_test_on}')
            for index in self.indexes_to_test_on:
                all_results[index['indexName']]  = self.client.index(index_name = index['indexName']).embed(
                    content=[
                        "Men shoes brown",
                        {"Large grey hat": 0.7, "https://marqo-assets.s3.amazonaws.com/tests/images/image1.jpg": 0.3}
                    ],
                    content_type=None
                )
            self.logger.debug(f"Ran prepare method for {self.indexes_to_test_on} inside test class {self.__class__.__name__}")
            self.save_results_to_file(all_results)
        except Exception as e: #TODO: This was called out as an antipattern last time - (logging & raising - fix it)
            self.logger.error(f"Exception occurred while embedding documents {e}")
            raise e

    def test_embed(self):
        self.logger.info(f"Running test_embed on {self.__class__.__name__}")
        stored_results = self.load_results_from_file()

        for index in self.indexes_to_test_on:
            index_name = index['indexName']
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



    def _compare_embed_results(self, expected_result, actual_result):
        self.assertEqual(expected_result.get("embeddings"), actual_result.get("embeddings"))
        self.assertEqual(expected_result.get("content"), actual_result.get("content"))