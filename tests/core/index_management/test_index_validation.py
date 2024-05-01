from pydantic import ValidationError
import pytest
import unittest
from marqo.core.index_management.index_management import IndexManagement


class TestIndexValidateSettings(unittest.TestCase):
    def generate_test_input(
        self, model="hf/e5-large", treat_urls_and_pointers_as_images=False
    ):
        # Given: Generate test data for the index
        test_data = {
            "treatUrlsAndPointersAsImages": treat_urls_and_pointers_as_images,
            "model": model,
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 2,
                "splitOverlap": 0,
                "splitMethod": "sentence",
            },
            "imagePreprocessing": {"patchMethod": None},
            "annParameters": {
                "spaceType": "euclidean",
                "parameters": {"efConstruction": 128, "m": 16},
            },
            "type": "unstructured",
        }
        return test_data

    def generate_invalid_test_input_snake_case(
        self, model="hf/e5-large", treat_urls_and_pointers_as_images=False
    ):
        # Given: Generate test data for the index
        test_data = {
            "treatUrlsAndPointersAsImages": treat_urls_and_pointers_as_images,
            "model": model,
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 2,
                "splitOverlap": 0,
                "splitMethod": "sentence",
            },
            "imagePreprocessing": {"patchMethod": None},
            "annParameters": {
                "spaceType": "euclidean",
                "parameters": {"efConstruction": 128, "m": 16},
            },
            "allFields": [
                {
                    "features": [
                        "lexical_search"
                    ],
                    "name": "title",
                    "type": "text"
                },
                {
                    "features": [
                        "lexical_search"
                    ],
                    "name": "description",
                    "type": "text"
                },
                {
                    "dependent_fields": {
                        "description": 0.3,
                        "title": 0.7
                    },
                    "name": "title_and_description",
                    "type": "multimodal_combination"
                }
            ],
            "type": "structured",
            "tensorFields": [
                "title_and_description"
            ]
        }
        return test_data

    def test_validate_index_settings_with_valid_text_based_input(self):
        # Given a valid input settings object
        input_settings = self.generate_test_input()

        # When validating the input settings object should not raise any exceptions
        (valid, error) = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validate_index_settings_with_valid_multimodal_based_input(self):
        input_settings = self.generate_test_input(
            model="open_clip/ViT-L-14/laion2b_s32b_b82k",
            treat_urls_and_pointers_as_images=True,
        )

        # When validating the input settings object should not raise any exceptions
        (valid, error) = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validate_index_settings_with_invalid_index_defaults(self):
        # Given an invalid input settings object (missing a required field)
        input_settings = {
            "model": "hf/e5-large",
            "normalizeEmbeddings": True,
            "numberOfShards": 5,
            "numberOfReplicas": 1,
        }

        # When validating the input settings object
        (valid, error) = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertFalse(valid)
        self.assertEqual(
            error,
            "2 validation errors for IndexSettings\n"
            "numberOfReplicas\n  extra fields not permitted (type=value_error.extra)\n"
            "numberOfShards\n  extra fields not permitted (type=value_error.extra)"
        )

    def test_validate_index_settings_with_invalid_snake_case_input(self):
        # Given an invalid input settings object (snake case)
        input_settings = {
            "dependent_fields": "value1",
            # other necessary fields as per your IndexSettings model
        }

        # When validating the input settings object
        (valid, error) = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertFalse(valid)
        self.assertEqual(
            error, 
            "1 validation error for IndexSettings\n"
            "__root__\n  Invalid field name 'dependent_fields'. See Create Index API reference "
            "here https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/ "
            "(type=value_error)"
        )
