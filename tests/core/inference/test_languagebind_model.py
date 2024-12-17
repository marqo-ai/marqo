import unittest

from pytest import mark

from marqo.core.inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.core.inference.embedding_models.languagebind_model_properties import *
from marqo.core.inference.image_download import format_and_load_CLIP_images
from marqo.s2_inference.s2_inference import _convert_vectorized_output
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search.streaming_media_processor import StreamingMediaProcessor
from tests.marqo_test import TestAudioUrls, TestImageUrls, TestVideoUrls


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

    def _help_test_encode_text_modality(self, model: LanguagebindModel, dimension = 768):
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
            self.assertEqual(len(converted_output ), len(test_case) if isinstance(test_case, list) else 1)
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

    def _help_test_encode_audio_modality(self, model, dimension = 768):
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
                url = audio, device="cuda", modality=Modality.AUDIO, 
                preprocessors=Preprocessors(**model.get_preprocessors()),
            )
            list_of_processed_audio.append(streaming_media_processor.process_media())
            
        test_cases.append(list_of_processed_audio)
        
        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.AUDIO)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def _help_test_encode_video_modality(self, model, dimension = 768):
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
            (list_of_processed_videos.append
             (streaming_media_processor.process_media()))
        test_cases.append(list_of_processed_videos)
        for test_case in test_cases:
            output = model.encode(test_case, modality=Modality.VIDEO)
            converted_output = _convert_vectorized_output(output)
            self.assertEqual(len(converted_output), len(test_case) if isinstance(test_case, list) else 1)
            for tensor in converted_output:
                self.assertEqual(dimension, len(tensor))

    def test_loading_languagebind_model_from_a_hf_repo(self):
        """A test for loading a LanguagebindModel from a Hugging Face repo."""
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