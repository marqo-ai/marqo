import unittest

import pytest
from pydantic import ValidationError

from marqo.core.index_management.index_management import IndexManagement


@pytest.mark.unittest
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

        # When validating the input settings object should not raise any exceptions but return None
        r = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertIsNone(r)

    def test_validate_index_settings_with_valid_multimodal_based_input(self):
        input_settings = self.generate_test_input(
            model="open_clip/ViT-L-14/laion2b_s32b_b82k",
            treat_urls_and_pointers_as_images=True,
        )

        # When validating the input settings object should not raise any exceptions but return None
        r = IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertIsNone(r)

    def test_validate_index_settings_with_invalid_index_defaults(self):
        # Given an invalid input settings object (missing a required field)
        input_settings = {
            "model": "hf/e5-large",
            "normalizeEmbeddings": True,
            "numberOfShards": 5,
            "numberOfReplicas": 1,
        }

        # When validating the input settings object
        with self.assertRaises(ValidationError) as context:
            IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertIn("2 validation errors for IndexSettings", str(context.exception))

    def test_validate_index_settings_with_invalid_snake_case_input(self):
        # Given an invalid input settings object (snake case)
        input_settings = {
            "dependent_fields": "value1",
            # other necessary fields as per your IndexSettings model
        }

        with self.assertRaises(ValidationError) as context:
            IndexManagement.validate_index_settings("test_index", input_settings)
        self.assertIn("__root__", str(context.exception))

        self.assertIn("Invalid field name 'dependent_fields'. See Create Index", 
                      str(context.exception))
