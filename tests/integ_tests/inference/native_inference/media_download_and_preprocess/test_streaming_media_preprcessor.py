import os.path
import time
import unittest
from unittest.mock import patch

import torch
from pytest import mark

from integ_tests.marqo_test import TestVideoUrls, TestAudioUrls
from marqo.core.inference.api import *
from marqo.inference.media_download_and_preprocess.streaming_media_processor import StreamingMediaProcessor
from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindPreprocessor
from marqo.s2_inference.errors import MediaDownloadError


class DummyPreprocessor(LanguagebindPreprocessor):
    def __init__(self, raw_preprocessor=None, device=None):
        super().__init__(raw_preprocessor, device)

    def preprocess(self, inputs, modality: Modality):
        if modality == Modality.TEXT:
            return inputs
        elif modality in (Modality.AUDIO, Modality.VIDEO, Modality.IMAGE):
            return [torch.rand(1, 10, 10) for _ in inputs]
        else:
            raise ValueError(f"Unsupported modality: {modality}")


class TestStreamingMediaProcessor(unittest.TestCase):
    """The integration test for StreamingMediaProcessor. You can include real downloading and preprocessing in this
    test class.
    """

    test_audio_preprocessing_config = AudioPreprocessingConfig(
        modality=Modality.AUDIO,
        download_thread_count=1,
        download_header=None,
        should_chunk=True,
        chunk_config=ChunkConfig(
            split_length=10,
            split_overlap=0
        )
    )

    test_video_preprocessing_config = VideoPreprocessingConfig(
        modality=Modality.VIDEO,
        download_thread_count=1,
        download_header=None,
        should_chunk=True,
        chunk_config=ChunkConfig(
            split_length=10,
            split_overlap=0
        )
    )

    test_preprocessor = DummyPreprocessor()

    def setUp(self):
        self.output_file = "./test.mp4"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_video_decode_cpu_works(self):
        """Video decoding should work on a CPU-only machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_video_gpu_acceleration = False

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url,  preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
            enable_video_gpu_acceleration=enable_video_gpu_acceleration
        )
        streaming_media_processor_object.fetch_video_chunk(start_time, duration, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

    def test_video_decode_cpu_invalid_url(self):
        """Invalid URL should raise a MediaDownloadError when instantiating the object and calls
        _fetch_file_metadata()."""
        invalid_url = "https://rqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        enable_video_gpu_acceleration = False

        with self.assertRaises(MediaDownloadError) as e:
            _ = StreamingMediaProcessor(
                url=invalid_url, preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config,
                enable_video_gpu_acceleration=enable_video_gpu_acceleration
            )

        self.assertFalse(os.path.exists(self.output_file))
        self.assertIn("404", str(e.exception))

    def test_video_decode_gpu_does_not_work(self):
        """A proper error is raised when trying to decode a video with GPU acceleration enabled on a CPU-only machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_video_gpu_acceleration = True

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
            enable_video_gpu_acceleration=enable_video_gpu_acceleration
        )

        with self.assertRaises(MediaDownloadError) as e:
            streaming_media_processor_object.fetch_video_chunk(start_time, duration, self.output_file)
        self.assertFalse(os.path.exists(self.output_file))

    @unittest.skip(reason="Temporarily skipped due to no support for languagebind model")
    @mark.largemodel
    def test_video_decode_cuda_works(self):
        """Both CPU and GPU decoding should work on a GPU-enabled machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 10

        for enable_video_gpu_acceleration in (True, False):
            streaming_media_processor_object = StreamingMediaProcessor(
                url=valid_url, preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config,
                enable_video_gpu_acceleration=enable_video_gpu_acceleration
            )

            decode_start_time = time.time()

            streaming_media_processor_object.fetch_video_chunk(
                start_time, duration, self.output_file
            )
            elapsed_time = time.time() - decode_start_time

            # We expect the GPU decoding to be faster than CPU decoding
            if enable_video_gpu_acceleration:
                self.assertLess(
                    elapsed_time, 3, f"GPU decoding took too long. Elapsed time: "
                                     f"{elapsed_time}. URL: {valid_url}"
                )
            self.assertTrue(os.path.exists(self.output_file))

    def test_audio_decode_cpu_works(self):
        """Audio decoding should work on a CPU-only machine."""
        valid_url = TestAudioUrls.AUDIO1.value
        start_time = 0
        duration = 1

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config
        )
        output_file = "./test.wav"
        streaming_media_processor_object.fetch_audio_chunk(start_time, duration, output_file)
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

    def test_metadata_fetching_success(self):
        """Metadata fetching should return correct size and duration."""
        valid_url = TestVideoUrls.VIDEO1.value

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config
        )
        size, duration = streaming_media_processor_object._fetch_file_metadata()

        self.assertEqual(2971504, size) # Hardcoded value
        self.assertEqual(10.01, duration) # Hardcoded value

    def test_metadata_fetching_invalid_url(self):
        """Invalid URL should raise MediaDownloadError when fetching metadata."""
        invalid_url = "https://invalid-url.com/video.mp4"

        with self.assertRaises(MediaDownloadError):
            streaming_media_processor_object = StreamingMediaProcessor(
                url=invalid_url, preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config
            )
            streaming_media_processor_object._fetch_file_metadata()

    def test_video_decoding_timeout(self):
        """Test that a timeout error is raised for slow video decoding."""
        valid_url = TestVideoUrls.VIDEO1.value
        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config
        )
        streaming_media_processor_object.VIDEO_CPU_TIMOUT_OUT_MULTIPLIER = 0.01  # Reduce timeout for testing

        with self.assertRaises(MediaDownloadError) as e:
            streaming_media_processor_object.fetch_video_chunk(0, 100, self.output_file)
        self.assertIn("timed out", str(e.exception))

    def test_header_conversion_with_empty_headers(self):
        """Empty headers should result in an empty string."""
        streaming_media_processor_object = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,  preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config
        )
        self.assertEqual(streaming_media_processor_object.media_download_header, "")

    def test_header_conversion_with_valid_headers(self):
        """Headers should be correctly converted to CLI format."""
        headers = {"Authorization": "Bearer token", "User-Agent": "Test"}
        test_video_preprocessing_config = VideoPreprocessingConfig(
            modality=Modality.VIDEO,
            download_thread_count=1,
            download_header=headers,
            should_chunk=True,
            chunk_config=ChunkConfig(
                split_length=10,
                split_overlap=0
            )
        )
        with patch("marqo.inference.media_download_and_preprocess"
                   ".streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata") \
                as mock_fetch_file_metadata:
            mock_fetch_file_metadata.return_value = (2971504, 10.01)
            streaming_media_processor_object = StreamingMediaProcessor(
                url=TestAudioUrls.AUDIO1.value, preprocessors=self.test_preprocessor,
                preprocessing_config=test_video_preprocessing_config
            )

        expected = "Authorization: Bearer token\r\nUser-Agent: Test"
        self.assertEqual(streaming_media_processor_object.media_download_header, expected)