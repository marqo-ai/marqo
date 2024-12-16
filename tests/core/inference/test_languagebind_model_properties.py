import unittest
from unittest import mock

import pytest
from pydantic import ValidationError

from marqo.core.inference.embedding_models.languagebind_model_properties import *
from marqo.tensor_search.models.external_apis.hf import HfModelLocation, HfAuth
from marqo.tensor_search.models.external_apis.s3 import S3Location, S3Auth
from marqo.tensor_search.models.private_models import ModelLocation


@pytest.mark.unittest
class TestLanguagebindModelProperties(unittest.TestCase):

    def test_supported_modalities(self):
        """A test for supported modalities in LanguagebindModelProperties.
        LANGUAGE modality is required, and at least one of the other modalities must be supported.
        """
        base_test_case = {
            "dimensions": 764,
            "type": "languagebind",
            "name": "test_model",
        }

        test_cases = [
            ({"supportedModalities": [Modality.LANGUAGE], **base_test_case}, "Must support one of audio, image, video"),
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

    def test_supported_modalities_can_accept_both_text_and_language(self):
        """Ensure that supported modalities can accept both 'text' and 'language'."""
        base_test_case = {
            "dimensions": 764,
            "type": "languagebind",
            "name": "test_model",
        }
        test_case = {
            "supportedModalities": [Modality.LANGUAGE, Modality.LANGUAGE],
            **base_test_case
        }
        properties = LanguagebindModelProperties(**test_case)
        self.assertEqual(properties.supportedModalities, [Modality.LANGUAGE, Modality.LANGUAGE])

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

        supported_modalities = [Modality.LANGUAGE, Modality.IMAGE]

        with self.assertRaises(ValueError) as context:
            LanguagebindModelProperties(
                modelLocation=language_bind_model_location,
                supportedModalities=supported_modalities,
                **base_test_case
            )
        self.assertIn("Mismatch between supported modalities and model location.",
                      str(context.exception))



    # def test_modality_location_invalid_url_and_auth(self):
    #     with self.assertRaises(ValueError) as context:
    #         ModalityLocation(url="http://example.com", authRequired=True)
    #     self.assertIn("authRequired must be False when url is provided", str(context.exception))
    #
    # def test_languagebind_model_location_valid(self):
    #     hf_auth = HfAuth(token="test_token")
    #     hf_location = HfModelLocation(
    #         repo_id="hf/test-model",
    #         filename="model.bin",
    #         auth=hf_auth
    #     )
    #     modality_location = ModalityLocation(hf=hf_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #     self.assertEqual(model_location.audio, modality_location)
    #
    # def test_languagebind_model_location_invalid_no_modality(self):
    #     with self.assertRaises(ValueError) as context:
    #         LanguagebindModelLocation()
    #     self.assertIn("At least one of audio, image, video must be provided", str(context.exception))
    #
    # def test_languagebind_model_properties_valid(self):
    #     hf_auth = HfAuth(token="test_token")
    #     hf_location = HfModelLocation(
    #         repo_id="hf/test-model",
    #         filename="model.bin",
    #         auth=hf_auth
    #     )
    #     modality_location = ModalityLocation(hf=hf_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #     properties = LanguagebindModelProperties(
    #         name="test_model",
    #         modelLocation=model_location,
    #         supportedModalities=[Modality.LANGUAGE],
    #         dimensions=768
    #     )
    #     self.assertEqual(properties.name, "test_model")
    #     self.assertEqual(properties.modelLocation, model_location)
    #
    # def test_languagebind_model_properties_invalid_name_and_model_location(self):
    #     hf_auth = HfAuth(token="test_token")
    #     hf_location = HfModelLocation(
    #         repo_id="hf/test-model",
    #         filename="model.bin",
    #         auth=hf_auth
    #     )
    #     modality_location = ModalityLocation(hf=hf_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #     with self.assertRaises(ValueError) as context:
    #         LanguagebindModelProperties(
    #             name="test_model",
    #             modelLocation=model_location,
    #             supportedModalities=[Modality.LANGUAGE],
    #             dimensions=768
    #         )
    #     self.assertIn("Only one of name or modelLocation must be provided", str(context.exception))
    #
    # def test_languagebind_model_properties_invalid_modalities(self):
    #     hf_auth = HfAuth(token="test_token")
    #     hf_location = HfModelLocation(
    #         repo_id="hf/test-model",
    #         filename="model.bin",
    #         auth=hf_auth
    #     )
    #     modality_location = ModalityLocation(hf=hf_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #     with self.assertRaises(ValueError) as context:
    #         LanguagebindModelProperties(
    #             modelLocation=model_location,
    #             supportedModalities=[]
    #         )
    #     self.assertIn("You model must include 'text' as a supported modality", str(context.exception))
    #
    # def test_languagebind_model_properties_valid_modalities_with_text(self):
    #     s3_auth = S3Auth(
    #         aws_secret_access_key="test_secret",
    #         aws_access_key_id="test_key"
    #     )
    #     s3_location = S3Location(
    #         Bucket="test-bucket",
    #         Key="model/key",
    #         auth=s3_auth
    #     )
    #     modality_location = ModalityLocation(s3=s3_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #     properties = LanguagebindModelProperties(
    #         modelLocation=model_location,
    #         supportedModalities=[Modality.LANGUAGE],
    #         dimensions=768
    #     )
    #     self.assertIn(Modality.LANGUAGE, properties.supportedModalities)
    #
    # def test_languagebind_model_properties_invalid_modality_without_location(self):
    #     hf_auth = HfAuth(token="test_token")
    #     hf_location = HfModelLocation(
    #         repo_id="hf/test-model",
    #         filename="model.bin",
    #         auth=hf_auth
    #     )
    #     modality_location = ModalityLocation(hf=hf_location, authRequired=True)
    #     model_location = LanguagebindModelLocation(audio=modality_location)
    #
    #     # Missing the 'image' modality in modelLocation while it's part of supportedModalities
    #     with self.assertRaises(ValueError) as context:
    #         LanguagebindModelProperties(
    #             modelLocation=model_location,
    #             supportedModalities=[Modality.LANGUAGE, Modality.IMAGE],  # Including 'text' as required
    #             dimensions=768
    #         )
    #     self.assertIn("The supported modality 'image' is not in the model location", str(context.exception))
