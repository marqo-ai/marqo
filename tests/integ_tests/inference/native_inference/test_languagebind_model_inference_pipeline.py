import os
from unittest.mock import patch

import numpy as np
import pytest

from marqo.core.inference.api import *
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from tests.integ_tests.inference.inference_test_case import InferenceTestCase
from tests.integ_tests.marqo_test import TestAudioUrls, TestImageUrls, TestVideoUrls


@pytest.mark.largemodel
class TestLanguagebindModelInferencePipeline(InferenceTestCase):
    """
    A test class for the LanguagebindModelInferencePipeline. All models should be tested under
    the CUDA environment, so we set the environment variable MARQO_MAX_CUDA_MODEL_MEMORY to 15.

    We test pipeline using the largest model 'LanguageBind/Video_V1.5_FT_Audio_FT_Image' as it can handle all modalities.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.device_patcher = patch.dict(
            os.environ, {"MARQO_MAX_CUDA_MODEL_MEMORY": "15"}
        )
        cls.device_patcher.start()
        cls.inference = NativeInferenceLocal(device_manager=DeviceManager())
        cls.model_name = "LanguageBind/Video_V1.5_FT_Audio_FT_Image"
        cls.model_properties = {
            "name": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
            "dimensions": 768,
            "type": "languagebind",
            "loader": "languagebind",
            "model_size": 8,
            "supported_modalities": ["video", "audio", "language", "image"],
            "video_chunk_length": 20,
            "audio_chunk_length": 10,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.device_patcher.stop()

    def test_languagebind_model_inference_pipeline_vectorise_text(self):
        """Test that the pipeline returns the embeddings for the two texts without chunking or prefix."""
        text_inference_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["text", "very long long long long text"],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
        )

        results = self.inference.vectorise(text_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual("text", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768,), results_2[0][1].shape)
        self.assertEqual("very long long long long text", results_2[0][0])

    def test_languagebind_model_inference_pipeline_vectorise_image(self):
        """Test that the pipeline returns the embeddings for the two images."""
        image_inference_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=[TestImageUrls.IMAGE1.value, TestImageUrls.IMAGE2.value],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=ImagePreprocessingConfig(
                should_chunk=False,
                download_thread_count=1,
            ),
        )

        results = self.inference.vectorise(image_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE1.value, results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768,), results_2[0][1].shape)
        self.assertEqual(TestImageUrls.IMAGE2.value, results_2[0][0])

    def test_languagebind_model_inference_pipeline_vectorise_audio_no_chunk(self):
        """Test that the pipeline returns the embeddings for the two audio files without chunking."""
        audio_inference_request = InferenceRequest(
            modality=Modality.AUDIO,
            contents=[
                TestAudioUrls.AUDIO1.value,  # 5 seconds
                TestAudioUrls.AUDIO2.value,  # Also 5 seconds
            ],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=AudioPreprocessingConfig(
                should_chunk=False,
                download_thread_count=1,
            ),
        )

        results = self.inference.vectorise(audio_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual("[0.0, 5.0]", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768,), results_2[0][1].shape)
        self.assertEqual("[0.0, 5.0]", results_2[0][0])

    def test_languagebind_model_inference_pipeline_vectorise_audio_with_chunk(self):
        """Test that the pipeline returns the embeddings for the one audio files with chunking."""
        audio_inference_request = InferenceRequest(
            modality=Modality.AUDIO,
            contents=[
                TestAudioUrls.AUDIO1.value,  # 5 seconds
            ],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=AudioPreprocessingConfig(
                should_chunk=True,
                download_thread_count=1,
                chunk_config=ChunkConfig(split_length=2, split_overlap=1),
            ),
        )

        results = self.inference.vectorise(audio_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 1)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertEqual(4, len(results_1))  # We should see 4 chunks

        expected_chunks = ["[0.0, 2.0]", "[1.0, 3.0]", "[2.0, 4.0]", "[3.0, 5.0]"]

        for i, chunk in enumerate(results_1):
            self.assertTrue(isinstance(chunk, tuple))
            self.assertTrue(isinstance(chunk[0], str))
            self.assertTrue(isinstance(chunk[1], np.ndarray))
            self.assertEqual((768,), chunk[1].shape)
            self.assertEqual(expected_chunks[i], chunk[0])

    def test_languagebind_model_inference_pipeline_vectorise_video_without_chunk(self):
        """Test that the pipeline returns the embeddings for the one two video files without chunking."""
        video_inference_request = InferenceRequest(
            modality=Modality.VIDEO,
            contents=[
                TestVideoUrls.VIDEO1.value,  # 10 seconds
                TestVideoUrls.VIDEO2.value,  # 10 seconds
            ],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=VideoPreprocessingConfig(
                should_chunk=False, download_thread_count=1
            ),
        )

        results = self.inference.vectorise(video_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 2)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertTrue(len(results_1) == 1)
        self.assertTrue(isinstance(results_1[0], tuple))
        self.assertTrue(isinstance(results_1[0][0], str))
        self.assertTrue(isinstance(results_1[0][1], np.ndarray))
        self.assertEqual((768,), results_1[0][1].shape)
        self.assertEqual("[0.0, 10.0]", results_1[0][0])

        results_2: list[tuple[str, ndarray]] = results.result[1]
        self.assertTrue(isinstance(results_2, list))
        self.assertTrue(len(results_2) == 1)
        self.assertTrue(isinstance(results_2[0], tuple))
        self.assertTrue(isinstance(results_2[0][0], str))
        self.assertTrue(isinstance(results_2[0][1], np.ndarray))
        self.assertEqual((768,), results_2[0][1].shape)
        self.assertEqual("[0.0, 10.0]", results_2[0][0])

    def test_languagebind_model_inference_pipeline_vectorise_video_with_chunk(self):
        """Test that the pipeline returns the embeddings for the one two video files with chunking."""
        video_inference_request = InferenceRequest(
            modality=Modality.VIDEO,
            contents=[
                TestVideoUrls.VIDEO1.value,  # 10 seconds
            ],
            device="cuda",
            model_config=ModelConfig(
                model_name=self.model_name,
                model_properties=self.model_properties,
                normalize_embeddings=True,
            ),
            preprocessing_config=VideoPreprocessingConfig(
                should_chunk=True,
                download_thread_count=1,
                chunk_config=ChunkConfig(split_length=8, split_overlap=2),
            ),
        )

        results = self.inference.vectorise(video_inference_request)
        self.assertTrue(isinstance(results, InferenceResult))
        self.assertTrue(isinstance(results.result, list))
        self.assertTrue(len(results.result) == 1)

        results_1: list[tuple[str, ndarray]] = results.result[0]
        self.assertTrue(isinstance(results_1, list))
        self.assertEqual(2, len(results_1))  # We should see 4 chunks

        expected_chunks = ["[0.0, 8.0]", "[2.0, 10.0]"]

        for i, chunk in enumerate(results_1):
            self.assertTrue(isinstance(chunk, tuple))
            self.assertTrue(isinstance(chunk[0], str))
            self.assertTrue(isinstance(chunk[1], np.ndarray))
            self.assertEqual((768,), chunk[1].shape)
            self.assertEqual(expected_chunks[i], chunk[0])
