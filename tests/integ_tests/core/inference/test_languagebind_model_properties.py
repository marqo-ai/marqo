import unittest

import pytest
from pydantic import ValidationError

from marqo.core.inference.api.modality import Modality
from marqo.inference.native_inference.embedding_models.languagebind_model_properties import *


@pytest.mark.unittest
class TestLanguagebindModelProperties(unittest.TestCase):

    def test_supported_modalities(self):
        """Test language/text modality is required and at least one of audio, image, video is provided."""
        base_test_case = {
            "dimensions": 764,
            "type": "languagebind",
            "name": "test_model",
        }

        test_cases = [
            ({"supportedModalities": [Modality.TEXT], **base_test_case}, "Must support one of audio, image, video"),
            ({"supportedModalities": [Modality.IMAGE], **base_test_case},  "Text modality is required"),
            ({"supportedModalities": ["text1"], **base_test_case}, "Invalid modality, "
                                                                   "must be one of 'text', 'image', 'audio', 'video'"),
            ({"supportedModalities": [None], **base_test_case}, "Invalid modality, can't be None"),
            ({"supportedModalities": [], **base_test_case}, "Invalid modality, can't be empty"),
            ({"supportedModalities": {}, **base_test_case}, "Invalid data type, must be a list"),
        ]

        for test_case, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValidationError) as context:
                    LanguagebindModelProperties(**test_case)
                self.assertIn("supported_modalities", str(context.exception))

    def test_name_or_model_location_must_be_provided(self):
        """A test for checking that only one of name or modelLocation must be provided."""
        base_test_case = {
            "dimensions": 764,
            "type": "languagebind",
            "name": "test_model",
        }
        test_cases = [
            ({**base_test_case}, "One of name or modelLocation must be provided. Neither provided"),
            ({"modelLocation": "test_model", "name": "test", **base_test_case},
             "Only one of name or modelLocation must be provided. Both provided"),
        ]
        for test_case, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValidationError) as context:
                    LanguagebindModelProperties(**test_case)
                self.assertIn("supported_modalities", str(context.exception))

    def test_modalities_match_model_location(self):
        """A test for checking that the supported modalities match the model location."""
        base_test_case = {
            "dimensions": 764,
            "type": "languagebind",
        }

        language_bind_model_location = LanguagebindModelLocation(
            audio=ModalityLocation(url = "http://example.com"),
        )
        supported_modalities = [Modality.TEXT, Modality.IMAGE]

        with self.assertRaises(ValueError) as context:
            LanguagebindModelProperties(
                modelLocation=language_bind_model_location,
                supportedModalities=supported_modalities,
                **base_test_case
            )
        self.assertIn("Mismatch between supported modalities and model location.",
                      str(context.exception))

    def test_valid_model_properties(self):
        """Test that valid model properties are accepted."""
        base_model_properties = {
            "dimensions": 764,
            "type": "languagebind",
        }

        valid_model_properties_list = [
            (
                {"supportedModalities": [Modality.TEXT, Modality.IMAGE],
                 "modelLocation": {"image": {"url": "http://example.com"}},
                 **base_model_properties}, "Custom model with image modality from URL"
            ),
            (
                {"supportedModalities": [Modality.TEXT, Modality.VIDEO],
                 "modelLocation": {"video": {"hf": {"repoId": "test_repo"}}},
                 **base_model_properties}, "Custom model with video modality from Hugging Face"
            ),
            (
                {"supportedModalities": [Modality.TEXT, Modality.VIDEO, Modality.IMAGE, Modality.AUDIO],
                 "modelLocation": {
                     "tokenizer": {"hf": {"repoId": "test_repo"}},
                     "video": {"url": "http://example.com/video"},
                     "image": {"s3": {"Bucket": "test_bucket", "Key": "test_key"}},
                     "audio": {"hf": {"repoId": "test_repo"}}},
                 **base_model_properties}, "A mixture of all modalities and tokenizers from different locations"
            ),
            (
                {"supportedModalities": [Modality.TEXT, Modality.IMAGE],
                 "name": "test_model",
                 **base_model_properties}, "Marqo registered model with name"
            )
        ]

        for model_properties, msg in valid_model_properties_list:
            with self.subTest(msg=msg):
                LanguagebindModelProperties(**model_properties)

    def test_model_location_exactly_one_field(self):
        """Test that exactly one of url, s3, or hf must be provided in ModalityLocation."""
        test_cases = [
            ({}, "No field provided"),
            ({"url": "http://example.com", "s3": S3Location(Bucket="test", Key="test")}, "Multiple fields provided"),
        ]

        for test_case, msg in test_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError) as context:
                    ModalityLocation(**test_case)
                self.assertIn("Exactly one of url, s3, hf must be provided", str(context.exception))

    def test_both_text_and_language_are_accepted_as_supported_modalities(self):
        """Test that both 'text' and 'language' are accepted as supported modalities and are mapped to 'language'."""
        base_model_properties = {
            "dimensions": 764,
            "type": "languagebind",
            "modelLocation": {"image": {"url": "http://example.com"}}
        }
        test_cases = [
            ({"supportedModalities": ["text", "image"], **base_model_properties}, "Text modality is accepted"),
            ({"supportedModalities": ["language", "image"], **base_model_properties}, "Language modality is accepted"),
        ]

        for model_properties, msg in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(
                    set(LanguagebindModelProperties(**model_properties).supportedModalities),
                    {Modality.TEXT, Modality.IMAGE}
                )
