import os.path
import time
import unittest
from unittest.mock import patch

import torch
from pytest import mark

from marqo.core.inference.api import *
from marqo.inference.media_download_and_preprocess.streaming_media_processor import (
    StreamingMediaProcessor,
)
from marqo.inference.native_inference.embedding_models.languagebind_model import (
    LanguagebindPreprocessor,
)
from tests.integ_tests.marqo_test import TestAudioUrls, TestImageUrls, TestVideoUrls


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
        chunk_config=ChunkConfig(split_length=10, split_overlap=0),
    )

    test_video_preprocessing_config = VideoPreprocessingConfig(
        modality=Modality.VIDEO,
        download_thread_count=1,
        download_header=None,
        should_chunk=True,
        chunk_config=ChunkConfig(split_length=10, split_overlap=0),
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
            url=valid_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
            enable_video_gpu_acceleration=enable_video_gpu_acceleration,
        )
        streaming_media_processor_object.fetch_video_chunk(
            start_time, duration, self.output_file
        )
        self.assertTrue(os.path.exists(self.output_file))

    def test_video_decode_cpu_invalid_url(self):
        """Invalid URL should raise a MediaDownloadError when instantiating the object and calls
        _fetch_file_metadata()."""
        invalid_url = "https://rqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        enable_video_gpu_acceleration = False

        with self.assertRaises(MediaDownloadError) as e:
            _ = StreamingMediaProcessor(
                url=invalid_url,
                preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config,
                enable_video_gpu_acceleration=enable_video_gpu_acceleration,
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
            url=valid_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
            enable_video_gpu_acceleration=enable_video_gpu_acceleration,
        )

        with self.assertRaises(MediaDownloadError) as e:
            streaming_media_processor_object.fetch_video_chunk(
                start_time, duration, self.output_file
            )
        self.assertFalse(os.path.exists(self.output_file))

    @mark.largemodel
    def test_video_decode_cuda_works(self):
        """Both CPU and GPU decoding should work on a GPU-enabled machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 10

        for enable_video_gpu_acceleration in (True, False):
            streaming_media_processor_object = StreamingMediaProcessor(
                url=valid_url,
                preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config,
                enable_video_gpu_acceleration=enable_video_gpu_acceleration,
            )

            decode_start_time = time.time()

            streaming_media_processor_object.fetch_video_chunk(
                start_time, duration, self.output_file
            )
            elapsed_time = time.time() - decode_start_time

            # We expect the GPU decoding to be faster than CPU decoding
            if enable_video_gpu_acceleration:
                self.assertLess(
                    elapsed_time,
                    3,
                    f"GPU decoding took too long. Elapsed time: "
                    f"{elapsed_time}. URL: {valid_url}",
                )
            self.assertTrue(os.path.exists(self.output_file))

    def test_audio_decode_cpu_works(self):
        """Audio decoding should work on a CPU-only machine."""
        valid_url = TestAudioUrls.AUDIO1.value
        start_time = 0
        duration = 1

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
        )
        output_file = "./test.wav"
        streaming_media_processor_object.fetch_audio_chunk(
            start_time, duration, output_file
        )
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

    def test_metadata_fetching_success(self):
        """Metadata fetching should return correct size and duration."""
        valid_url = TestVideoUrls.VIDEO1.value

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
        )
        size, duration, _ = streaming_media_processor_object._fetch_file_metadata()

        self.assertEqual(2971504, size)  # Hardcoded value
        self.assertEqual(10.01, duration)  # Hardcoded value

    def test_metadata_fetching_invalid_url(self):
        """Invalid URL should raise MediaDownloadError when fetching metadata."""
        invalid_url = "https://invalid-url.com/video.mp4"

        with self.assertRaises(MediaDownloadError):
            streaming_media_processor_object = StreamingMediaProcessor(
                url=invalid_url,
                preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_video_preprocessing_config,
            )

    def test_video_decoding_timeout(self):
        """Test that a timeout error is raised for slow video decoding."""
        valid_url = TestVideoUrls.VIDEO1.value
        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_video_preprocessing_config,
        )
        streaming_media_processor_object.VIDEO_CPU_TIMOUT_OUT_MULTIPLIER = (
            0.01  # Reduce timeout for testing
        )

        with self.assertRaises(MediaDownloadError) as e:
            streaming_media_processor_object.fetch_video_chunk(0, 100, self.output_file)
        self.assertIn("timed out", str(e.exception))

    def test_header_conversion_with_empty_headers(self):
        """Empty headers should result in an empty string."""
        streaming_media_processor_object = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
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
            chunk_config=ChunkConfig(split_length=10, split_overlap=0),
        )
        with patch(
            "marqo.inference.media_download_and_preprocess"
            ".streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
        ) as mock_fetch_file_metadata:
            mock_fetch_file_metadata.return_value = (2971504, 10.01, Modality.VIDEO)
            streaming_media_processor_object = StreamingMediaProcessor(
                url=TestVideoUrls.VIDEO1.value,
                preprocessors=self.test_preprocessor,
                preprocessing_config=test_video_preprocessing_config,
            )

        expected = "Authorization: Bearer token\r\nUser-Agent: Test"
        self.assertEqual(
            streaming_media_processor_object.media_download_header, expected
        )

    def test_prob_modality_correct_video(self):
        for url in [
            TestVideoUrls.VIDEO1.value,
            TestVideoUrls.VIDEO2.value,
            TestVideoUrls.VIDEO3.value,
            TestVideoUrls.MKV_VIDEO1.value,
            TestVideoUrls.WEBM_VIDEO1.value,
            TestVideoUrls.AVI_VIDEO1.value,
        ]:
            with self.subTest(url=url):
                streaming_media_processor_object = StreamingMediaProcessor(
                    url=url,
                    preprocessors=self.test_preprocessor,
                    preprocessing_config=self.test_video_preprocessing_config,
                )
                self.assertEqual(
                    Modality.VIDEO, streaming_media_processor_object.probed_modality
                )

    def test_prob_modality_correct_audio(self):
        for url in [
            TestAudioUrls.AUDIO1.value,
            TestAudioUrls.AUDIO2.value,
            TestAudioUrls.AUDIO3.value,
            TestAudioUrls.MP3_AUDIO1.value,
            TestAudioUrls.MP3_AUDIO1.value,
            TestAudioUrls.ACC_AUDIO1.value,
            TestAudioUrls.OGG_AUDIO1.value,
            TestAudioUrls.FLAC_AUDIO1.value,
        ]:
            with self.subTest(url=url):
                streaming_media_processor_object = StreamingMediaProcessor(
                    url=url,
                    preprocessors=self.test_preprocessor,
                    preprocessing_config=self.test_audio_preprocessing_config,
                )
                self.assertEqual(
                    Modality.AUDIO, streaming_media_processor_object.probed_modality
                )

    def test_prob_modality_correct_image(self):
        """Ensure that the probed modality is correct for various image formats. Note that
        an error is raised as StreamingMediaProcessor is not designed to handle images."""
        for url in [
            TestImageUrls.IMAGE1.value,
            TestImageUrls.IMAGE2.value,
            TestImageUrls.IMAGE3.value,
            TestImageUrls.COCO.value,
        ]:
            with self.subTest(url=url):
                with self.assertRaises(MediaMismatchError) as e:
                    _ = StreamingMediaProcessor(
                        url=url,
                        preprocessors=self.test_preprocessor,
                        preprocessing_config=self.test_video_preprocessing_config,
                    )
                self.assertIn("the detected modality image", str(e.exception))

    def test_incorrect_modality_between_audio_and_video_will_raise_an_error(self):
        test_cases = [
            (
                TestVideoUrls.VIDEO1.value,
                self.test_audio_preprocessing_config,
                "The url is video, but the preprocessing config is audio",
            ),
            (
                TestAudioUrls.AUDIO1.value,
                self.test_video_preprocessing_config,
                "The url is audio, but the preprocessing config is video",
            ),
        ]
        for url, processing_config, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(MediaMismatchError) as e:
                    _ = StreamingMediaProcessor(
                        url=url,
                        preprocessors=self.test_preprocessor,
                        preprocessing_config=processing_config,
                    )
                self.assertIn(
                    "Please check your media file and try again", str(e.exception)
                )

    def test_video_chunk_generated_when_duration_smaller_than_overlap(self):
        audio_url = TestVideoUrls.VIDEO1.value  # duration is 10s
        preprocessing_config = self.test_video_preprocessing_config.copy(
            update={"chunk_config": ChunkConfig(split_length=20, split_overlap=11)}
        )
        streaming_media_processor_object = StreamingMediaProcessor(
            url=audio_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=preprocessing_config,
        )

        processed_chunks = streaming_media_processor_object.process_media()
        self.assertEqual(1, len(processed_chunks))
        self.assertEqual("[0.0, 10.0]", processed_chunks[0][0])

    def test_audio_chunk_generated_when_duration_smaller_than_overlap(self):
        audio_url = TestAudioUrls.AUDIO1.value  # duration is 5s
        preprocessing_config = self.test_audio_preprocessing_config.copy(
            update={"chunk_config": ChunkConfig(split_length=10, split_overlap=6)}
        )
        streaming_media_processor_object = StreamingMediaProcessor(
            url=audio_url,
            preprocessors=self.test_preprocessor,
            preprocessing_config=preprocessing_config,
        )

        processed_chunks = streaming_media_processor_object.process_media()
        self.assertEqual(1, len(processed_chunks))
        self.assertEqual("[0.0, 5.0]", processed_chunks[0][0])
