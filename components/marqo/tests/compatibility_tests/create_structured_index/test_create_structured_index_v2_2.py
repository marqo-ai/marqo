import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.2.0')
class TestCreateStructuredIndexv2_2(BaseCompatibilityTestCase):
    """
    New structured index data types: long, double, array<long> and array<double> for a higher precision and range of values Available for indexes created with Marqo 2.2+ (#722)
    Ref: https://github.com/marqo-ai/marqo/releases/tag/2.2.0
    """
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
                {"name": "boolean_field", "type": "bool"},
                {"name": "float_field_1", "type": "float"},
                {"name": "array_int_field_1", "type": "array<int>"},
                {"name": "array_float_field_1", "type": "array<float>"},
                {"name": "array_long_field_1", "type": "array<long>"},
                {"name": "array_double_field_1", "type": "array<double>"},
                {"name": "long_field_1", "type": "long"},
                {"name": "double_field_1", "type": "double"},
            ],
            "tensorFields": ["multimodal_field"],
            "annParameters": {
                "spaceType": "prenormalized-angular",
                "parameters": {"efConstruction": 512, "m": 16},
            }
        }]
    indexes_to_test_on = ["test_create_index_api_structured_index_2_2_0"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().setUpClass()

    def prepare(self):
        self.logger.debug(f"Creating indexes {self.indexes_settings_to_test_on}")
        all_results = {}
        errors = []  # Collect any errors to report them at the end
        for index_name, index_settings in zip(self.indexes_to_test_on, self.indexes_settings_to_test_on):
            try:
                self.client.create_index(index_name, settings_dict = index_settings)
                all_results[index_name] = self.client.index(index_name).get_settings()
            except Exception as e:
                errors.append((index_name, traceback.format_exc()))

        if errors:
            formatted_errors = [
                f"Index: {index_name}\nTraceback:\n{error}"
                for index_name, error in errors
            ]
            self.logger.error("\n".join(formatted_errors))

        self.save_results_to_file(all_results)

    def test_expected_settings(self):
        expected_settings = self.load_results_from_file()
        test_failures = [] # this stores the failures in the subtests. These failures could be assertion errors or any other types of exceptions

        for index_name in self.indexes_to_test_on:
            try:
                with self.subTest(index = index_name):
                    actual_settings = self.client.index(index_name).get_settings()
                    self.logger.debug(f"Printing actual_settings {actual_settings}")
                    self.logger.debug(f"Printing expected_settings {expected_settings.get(index_name)}")
                    self.assertEqual(expected_settings.get(index_name), actual_settings, f"Index settings do not match expected settings, expected {expected_settings}, but got {actual_settings}")
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        # After all subtests, raise a comprehensive failure if any occurred
        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")