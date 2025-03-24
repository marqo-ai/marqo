import os
import unittest
from unittest.mock import patch

from pytest import mark

from integ_tests.marqo_test import TestAudioUrls, TestImageUrls, TestVideoUrls
from marqo.core.inference.api.modality import Modality
from marqo.inference.media_download_and_preprocess.image_download import format_and_load_CLIP_images
from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.s2_inference.s2_inference import _convert_vectorized_output
from marqo.tensor_search.models.external_apis.hf import HfAuth
from marqo.tensor_search.models.external_apis.s3 import S3Auth
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.streaming_media_processor import StreamingMediaProcessor


@unittest.skip(reason="Temporarily skipped due to no support for languagebind model")
@mark.unittest
@mark.largemodel
class TestLanguagebindModels(unittest.TestCase):
    """
    A test class for the LanguagebindModel class. These are all unit tests that does not require connection
    to the vector database.
    """
    AUDIO_HF_REPO_NAME = "Marqo/LanguageBind_Audio_FT"
    IMAGE_HF_REPO_NAME = "Marqo/LanguageBind_Image"
    VIDEO_HF_REPO_NAME = "Marqo/LanguageBind_Video_V1.5_FT"
    PRIVATE_VIDEO_HF_REPO = "Marqo/private-LanguageBind_Video_V1.5_FT"
    AUDIO_URL = "https://opensource-languagebind-models.s3.us-east-1.amazonaws.com/LanguageBind_Audio_FT.zip"

    aws_access_key_id = os.getenv("PRIVATE_MODEL_TESTS_AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.getenv("PRIVATE_MODEL_TESTS_AWS_SECRET_ACCESS_KEY", None)
    hf_token = os.getenv("PRIVATE_MODEL_TESTS_HF_TOKEN", None)

    def _help_test_encode_text_modality(self, model: LanguagebindModel, dimension=768):
        """A helper function for testing the encode method for text modality.

        The Languagebind model should be able to encode text in the following formats:
        1. - A single string > from search, add_documents
        2. - A list of strings > from weighted search, add_documents
        """
        test_cases = [
            "test", ["simple test", "simple test"]
        ]

        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.TEXT)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def _help_test_encode_image_modality(self, model: LanguagebindModel, dimension=768):
        """A helper function for testing the encode method for image modality.

        The languagebind model should be able to encode images in the following formats:
        - An URL of an image > from search
        - A List of URLs of images > from weighted search
        - A List of preprocessed images > from add_documents
        """
        test_cases = [
            TestImageUrls.IMAGE2.value,
            [TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value],
        ]
        list_of_pil_images = format_and_load_CLIP_images(
            [TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value], dict())

        list_of_processed_image = [
            model.get_preprocessors()[Modality.IMAGE.value](image, return_tensors='pt') for image in list_of_pil_images
        ]

        test_cases.append(list_of_processed_image)
        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.IMAGE)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def _help_test_encode_audio_modality(self, model, dimension=768):
        """A helper function for testing the encode method for audio modality.

        The languagebind model should be able to encode images in the following formats:
        - An URL of an audio > from search
        - A List of URLs of audios > from weighted search
        - A List of preprocessed audios > from add_documents
        """
        test_cases = [
            TestAudioUrls.AUDIO1.value,
            [TestAudioUrls.AUDIO2.value, TestAudioUrls.AUDIO3.value]
        ]

        list_of_audios = [TestAudioUrls.AUDIO2.value, TestAudioUrls.AUDIO3.value]
        list_of_processed_audio = []
        for audio in list_of_audios:
            streaming_media_processor = StreamingMediaProcessor(
                url=audio, device="cuda", modality=Modality.AUDIO,
                preprocessors=Preprocessors(**model.get_preprocessors()),
            )
            list_of_processed_audio.append(streaming_media_processor.process_media()[0]["tensor"])

        test_cases.append(list_of_processed_audio)

        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.AUDIO)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def _help_test_encode_video_modality(self, model, dimension=768):
        """A helper function for testing the encode method for video modality.

        The languagebind model should be able to encode images in the following formats:
        - An URL of a video > from search
        - A List of URLs of videos > from weighted search
        - A List of preprocessed videos > from add_documents
        """
        test_cases = [
            TestVideoUrls.VIDEO1.value,
            [TestVideoUrls.VIDEO2.value, TestVideoUrls.VIDEO3.value]
        ]

        list_of_videos = [TestVideoUrls.VIDEO2.value, TestVideoUrls.VIDEO3.value]
        list_of_processed_videos = []
        for audio in list_of_videos:
            streaming_media_processor = StreamingMediaProcessor(
                url=audio, device="cuda", modality=Modality.VIDEO,
                preprocessors=Preprocessors(**model.get_preprocessors()),
                enable_video_gpu_acceleration=True
            )
            list_of_processed_videos.append(streaming_media_processor.process_media()[0]["tensor"])
        test_cases.append(list_of_processed_videos)
        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.VIDEO)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def test_loading_languagebind_model_from_a_hf_repo(self):
        """A test for loading a LanguagebindModel from a public Hugging Face repo."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"hf": {"repoId": self.IMAGE_HF_REPO_NAME}},
                "audio": {"hf": {"repoId": self.AUDIO_HF_REPO_NAME}},
                "video": {"hf": {"repoId": self.VIDEO_HF_REPO_NAME}}
            }
        }

        model = LanguagebindModel(device="cuda", model_properties=model_properties)
        model.load()

        test_cases = [
            (self._help_test_encode_text_modality, "Test text modality"),
            (self._help_test_encode_image_modality, "Test image modality"),
            (self._help_test_encode_audio_modality, "Test audio modality"),
            (self._help_test_encode_video_modality, "Test video modality")
        ]

        for test_case, msg, in test_cases:
            with self.subTest(msg=msg):
                test_case(model)

    def test_loading_languagebind_model_from_a_private_hf_repo(self):
        """A test for loading a LanguagebindModel from a private Hugging Face repo."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "video"],
            "modelLocation": {
                "video": {"hf": {"repoId": self.PRIVATE_VIDEO_HF_REPO}}
            }
        }

        mode_auth = ModelAuth(hf=HfAuth(token=self.hf_token))

        model = LanguagebindModel(device="cuda", model_properties=model_properties, model_auth=mode_auth)
        model.load()

        test_cases = [
            (self._help_test_encode_text_modality, "Test text modality"),
            (self._help_test_encode_video_modality, "Test video modality")
        ]

        for test_case, msg, in test_cases:
            with self.subTest(msg=msg):
                test_case(model)

    def test_loading_languagebind_model_from_a_url(self):
        """A test for loading a LanguagebindModel from a URL."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "audio"],
            "modelLocation": {
                "audio": {"url": self.AUDIO_URL}
            }
        }

        model = LanguagebindModel(device="cuda", model_properties=model_properties)
        model.load()

        test_cases = [
            (self._help_test_encode_text_modality, "Test text modality"),
            (self._help_test_encode_audio_modality, "Test audio modality")
        ]

        for test_case, msg, in test_cases:
            with self.subTest(msg=msg):
                test_case(model)

    def test_loading_languagebind_model_from_a_zip_on_s3(self):
        """A test for loading a LanguagebindModel from a zip file on S3."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Image.zip"}},
                "audio": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Audio_FT.zip"}},
                "video": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Video_V1.5_FT.zip"}},
            }
        }

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_secret_access_key=self.aws_secret_access_key,
                aws_access_key_id=self.aws_access_key_id)
        )
        model = LanguagebindModel(
            device="cuda", model_properties=model_properties, model_auth=model_auth)
        model.load()

        test_cases = [
            (self._help_test_encode_text_modality, "Test text modality"),
            (self._help_test_encode_image_modality, "Test image modality"),
            (self._help_test_encode_audio_modality, "Test audio modality"),
            (self._help_test_encode_video_modality, "Test video modality")
        ]

        for test_case, msg, in test_cases:
            with self.subTest(msg=msg):
                test_case(model)

    def test_loading_languagebind_model_from_a_zip_on_s3_with_role(self):
        """A test for loading a LanguagebindModel from a zip file on S3 using a role."""
        model_properties = {
            "dimensions": 768,
            "type": "languagebind",
            "supportedModalities": ["text", "image", "audio", "video"],
            "modelLocation": {
                "image": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Image.zip"}},
                "audio": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Audio_FT.zip"}},
                "video": {"s3": {"Bucket": "opensource-languagebind-models", "Key": "LanguageBind_Video_V1.5_FT.zip"}},
            }
        }

        model = LanguagebindModel(
            device="cuda", model_properties=model_properties
        )
        raised_exception = RuntimeError("Stop here")
        with (patch("marqo.inference.model_download.model_download.get_presigned_s3_url",side_effect=raised_exception)
              as mock_presigned_url):
            with patch("marqo.inference.model_download.model_download.check_s3_model_already_exists", return_value=False):
                with self.assertRaises(RuntimeError) as context:
                    model.load()

        # Ensure that the get_presigned_s3_url function was called thus role based access was attempted
        mock_presigned_url.assert_called_once()