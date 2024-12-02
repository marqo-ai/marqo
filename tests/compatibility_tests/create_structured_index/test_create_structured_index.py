import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.0.0')
class TestCreateStructuredIndex(BaseCompatibilityTestCase):
    indexes_settings_to_test_on = [
        {
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
                {"name": "caption", "type": "text", "features": ["lexical_search", "filter"]},
                {"name": "tags", "type": "array<text>", "features": ["filter"]},
                {"name": "image_field", "type": "image_pointer"},
                {"name": "my_int", "type": "int", "features": ["score_modifier"]},
                # this field maps the above image field and text fields into a multimodal combination.
                {
                    "name": "multimodal_field",
                    "type": "multimodal_combination",
                    "dependentFields": {"image_field": 0.9, "text_field": 0.1},
                },
            ],
            "tensorFields": ["multimodal_field"],
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {"efConstruction": 512, "m": 16},
            }
        }]
    indexes_to_test_on = ["test_create_index_api_structured_index"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def prepare(self):
        self.logger.info(f"Creating indexes {self.indexes_settings_to_test_on}")
        for index_name, index_settings in zip(self.indexes_to_test_on, self.indexes_settings_to_test_on):
            self.client.create_index(index_name, settings_dict = index_settings)

    def test_expected_settings(self):
        # expected_settings = self.indexes_settings_to_test_on
        for index_name, expected_settings in zip(self.indexes_to_test_on, self.indexes_settings_to_test_on):
            try:
                actual_settings = self.client.index(index_name).get_settings()
                self.logger.debug(f"Printing actual_settings {actual_settings}")
                self.logger.debug(f"Printing expected_settings {expected_settings}")
            except Exception as e:
                self.logger.error(f"Exception while getting index settings {e}")
                raise e
            self.assertEqual(expected_settings, actual_settings)