from pydantic import ValidationError
import pytest
import json
import unittest

from fastapi.responses import JSONResponse

from marqo.core.index_management.validation import validate_settings_object


class TestValidateSettings(unittest.TestCase):
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
        return json.dumps(test_data)

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
        return json.dumps(test_data)

    def test_validate_settings_object_with_valid_text_based_input(self):
        # Given a valid input settings object
        input_settings = self.generate_test_input()

        # When validating the input settings object
        result = validate_settings_object("test_index", input_settings)
        self.assertTrue(result)

    def test_validate_settings_object_with_valid_multimodal_based_input(self):
        input_settings = self.generate_test_input(
            model="open_clip/ViT-L-14/laion2b_s32b_b82k",
            treat_urls_and_pointers_as_images=True,
        )

        result = validate_settings_object("test_index", input_settings)
        self.assertTrue(result)

    def test_validate_settings_object_with_invalid_index_defaults(self):
        # Given an invalid input settings object (missing a required field)
        input_settings = {
            "model": "hf/e5-large",
            "normalizeEmbeddings": True,
            "numberOfShards": 5,
            "numberOfReplicas": 1,
        }

        # When validating the input settings object
        with pytest.raises(ValidationError) as exc_info:
            validate_settings_object("test_index", json.dumps(input_settings))

        # Check the exception details
        assert str(exc_info.value) == (
            "2 validation errors for IndexSettings\n"
            "numberOfReplicas\n  extra fields not permitted (type=value_error.extra)\n"
            "numberOfShards\n  extra fields not permitted (type=value_error.extra)"
        ), "Expected validation errors did not match."

    def test_validate_settings_object_with_invalid_snake_case_input(self):
        # Given an invalid input settings object (snake case)
        input_settings = {
            "dependent_fields": "value1",
            # other necessary fields as per your IndexSettings model
        }

        # When validating the input settings object
        with pytest.raises(ValidationError) as exc_info:
            validate_settings_object("test_index", json.dumps(input_settings))

        # Check the exception details
        assert str(exc_info.value) == (
            "1 validation error for IndexSettings\n"
            "__root__\n  Invalid field name 'dependent_fields'. See Create Index API reference "
            "here https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/ "
            "(type=value_error)"
        ), "Expected validation errors did not match."

    def assert_json_response_equal(self, response1: JSONResponse, response2: JSONResponse):
        self.assertEqual(response1.status_code,
                         response2.status_code, "Status codes differ")
        self.assertEqual(response1.body, response2.body,
                         "Response bodies differ")
