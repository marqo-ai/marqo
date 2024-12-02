import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
class TestCreateIndex(BaseCompatibilityTestCase):
    index_name = "test_create_index_api"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = [cls.index_name]
        super().tearDownClass()

    def prepare(self):

        try:
            self.client.create_index(index_name = self.index_name)
        except Exception as e:
            raise Exception(f"Exception when creating index with name {self.index_name}")

    def test_expected_settings(self):
        expected_settings = {
            'type': 'unstructured',
            'treatUrlsAndPointersAsImages': False,
            'treatUrlsAndPointersAsMedia': False,
            'filterStringMaxLength': 50,
            'model': 'hf/e5-base-v2',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'splitLength': 2, 'splitOverlap': 0, 'splitMethod': 'sentence'},
            'imagePreprocessing': {},
            'audioPreprocessing': {'splitLength': 10, 'splitOverlap': 3},
            'videoPreprocessing': {'splitLength': 20, 'splitOverlap': 3},
            'vectorNumericType': 'float',
            'annParameters': {
                'spaceType': 'prenormalized-angular', 'parameters': {
                    'efConstruction': 512, 'm': 16}
            }
        }
        try:
            actual_settings = self.client.index(self.index_name).get_settings()
        except Exception as e:
            raise Exception(f"Exception when getting index settings for index {self.index_name}")

        self.logger.debug(f"Expected settings: {expected_settings}")
        self.assertEqual(expected_settings, actual_settings, f"Index settings do not match expected settings, expected {expected_settings}, but got {actual_settings}")