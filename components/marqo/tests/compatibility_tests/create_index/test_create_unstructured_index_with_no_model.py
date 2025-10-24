from sys import exc_info

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.12.0')
class TestCreateIndexWithNoModel(BaseCompatibilityTestCase):
    index_name = "test_create_index_api_no_model"
    settings = {
        "treatUrlsAndPointersAsImages": False,
        "model": "no_model",
        "modelProperties": {
            "dimensions": 384,  # Set the dimensions of the vectors
            "type": "no_model"  # This is required
        }
    }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [cls.index_name]
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = [cls.index_name]
        super().setUpClass()
    
    def prepare(self):

        all_results = {}
        try:
            self.logger.debug(f"Creating index {self.index_name}")
            self.client.create_index(index_name = self.index_name, settings_dict = self.settings)
            all_results[self.index_name] = self.client.index(self.index_name).get_settings()
            self.save_results_to_file(all_results)
        except Exception as e:
            raise Exception(f"Exception when creating index with name {self.index_name}") from e

    def test_expected_settings(self):
        try:
            expected_settings = self.load_results_from_file()
            actual_settings = self.client.index(self.index_name).get_settings()
        except Exception as e:
            raise Exception(f"Exception when getting index settings for index {self.index_name}") from e

        self.logger.debug(f"Expected settings: {expected_settings}")
        self.logger.debug(f"Actual settings: {actual_settings}")
        self.assertEqual(expected_settings[self.index_name], actual_settings, f"Index settings do not match expected settings, expected {expected_settings}, but got {actual_settings}")