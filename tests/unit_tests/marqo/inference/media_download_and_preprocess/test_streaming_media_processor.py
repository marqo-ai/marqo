import os.path
import unittest
from unittest.mock import patch

import ffmpeg
import torch

from marqo.core.inference.api import *
from marqo.inference.media_download_and_preprocess.streaming_media_processor import (
    ChunkTimingGenerator,
    StreamingMediaProcessor,
)
from marqo.inference.native_inference.embedding_models.languagebind_model import (
    LanguagebindPreprocessor,
)
from tests.integ_tests.marqo_test import TestAudioUrls, TestVideoUrls


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
    """The unit test class for StreamingMediaProcessor.

    We should avoid real video/audio downloading and decoding in unit tests. You can do the long-time tests in
    the integration tests.
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

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_process_audio_media_chunks(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (
            1000000,
            30.0,
            Modality.AUDIO,
        )  # size, duration
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        processor = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
        )

        processed_chunks = processor.process_media()
        self.assertEqual(len(processed_chunks), 3)  # 30 seconds / 10 seconds per chunk
        self.assertTrue(
            all(isinstance(tensor, torch.Tensor) for _, tensor in processed_chunks)
        )

    @patch("ffmpeg.probe")
    def test_fetch_file_metadata_failure(self, mock_ffmpeg_probe):
        mock_ffmpeg_probe.side_effect = ffmpeg.Error("ffmpeg", b"", b"error")
        with self.assertRaises(MediaDownloadError) as context:
            processor = StreamingMediaProcessor(
                url=TestAudioUrls.AUDIO1.value,
                preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_audio_preprocessing_config,
            )

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_last_chunk_alignment(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (
            1000000,
            23.5,
            Modality.AUDIO,
        )  # Non-divisible duration
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        processor = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
        )

        processed_chunks = processor.process_media()
        last_chunk_time, _ = processed_chunks[-1]
        start_time, end_time = map(float, last_chunk_time.strip("[]").split(","))

        # Check that the last chunk ends at the media duration
        self.assertAlmostEqual(end_time, 23.5, places=2)
        # Check that the last chunk starts at max(duration - split_length, 0)
        self.assertAlmostEqual(
            start_time,
            max(
                23.5 - self.test_audio_preprocessing_config.chunk_config.split_length, 0
            ),
            places=2,
        )

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_explicit_audio_chunk_times(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        # Set up media duration exactly 30s
        mock_fetch_file_metadata.return_value = (1000000, 30.0, Modality.AUDIO)
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        overlap_config = AudioPreprocessingConfig(
            modality=Modality.AUDIO,
            download_thread_count=1,
            download_header=None,
            should_chunk=True,
            chunk_config=ChunkConfig(split_length=10, split_overlap=2),
        )

        processor = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=overlap_config,
        )

        processed_chunks = processor.process_media()

        # Define the exact expected chunk times
        expected_chunk_times = [
            (0.0, 10.0),  # First chunk
            (8.0, 18.0),  # Second chunk overlaps with the first
            (16.0, 26.0),  # Third chunk overlaps with the second
            (20.0, 30.0),  # Last chunk, note that it ends at 30.0 but starts at 20.0
        ]

        self.assertEqual(len(expected_chunk_times), len(processed_chunks))

        for (chunk_time_str, tensor), (expected_start, expected_end) in zip(
            processed_chunks, expected_chunk_times
        ):
            # Validate chunk time format and values
            self.assertTrue(
                chunk_time_str.startswith("[") and chunk_time_str.endswith("]")
            )
            start_time, end_time = map(float, chunk_time_str.strip("[]").split(","))

            self.assertAlmostEqual(start_time, expected_start, places=2)
            self.assertAlmostEqual(end_time, expected_end, places=2)

            # Validate tensor correctness
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, torch.Size([1, 10, 10]))

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_short_media_duration(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (
            1000000,
            5.0,
            Modality.AUDIO,
        )  # Shorter than split_length
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        processor = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
        )

        processed_chunks = processor.process_media()
        self.assertEqual(len(processed_chunks), 1)

        chunk_time, tensor = processed_chunks[0]
        start_time, end_time = map(float, chunk_time.strip("[]").split(","))
        self.assertAlmostEqual(start_time, 0.0, places=2)
        self.assertAlmostEqual(end_time, 5.0, places=2)

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_media_download_error_is_raised(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (1000000, 10.0, Modality.AUDIO)
        mock_fetch_audio_chunk.side_effect = MediaDownloadError(
            "Failed downloading audio"
        )

        processor = StreamingMediaProcessor(
            url=TestAudioUrls.AUDIO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=self.test_audio_preprocessing_config,
        )
        with self.assertRaises(MediaDownloadError) as context:
            result = processor.process_media()

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_audio_chunk"
    )
    def test_media_exceeds_max_size_error_is_raised(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (1e10, 10.0, Modality.AUDIO)
        mock_fetch_audio_chunk.side_effect = MediaDownloadError(
            "Failed downloading audio"
        )

        with self.assertRaises(MediaExceedsMaxSizeError) as context:
            processor = StreamingMediaProcessor(
                url=TestAudioUrls.AUDIO1.value,
                preprocessors=self.test_preprocessor,
                preprocessing_config=self.test_audio_preprocessing_config,
            )

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_video_chunk"
    )
    def test_explicit_video_chunk_times_with_overlaps(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (1000000, 41.0, Modality.VIDEO)
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        overlap_config = VideoPreprocessingConfig(
            modality=Modality.VIDEO,
            download_thread_count=1,
            download_header=None,
            should_chunk=True,
            chunk_config=ChunkConfig(split_length=20, split_overlap=2),
        )

        processor = StreamingMediaProcessor(
            url=TestVideoUrls.VIDEO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=overlap_config,
        )

        processed_chunks = processor.process_media()

        # Define the exact expected chunk times
        expected_chunk_times = [
            (0.0, 20.0),  # First chunk
            (18.0, 38.0),  # Second chunk overlaps with the first
            (
                21.0,
                41.0,
            ),  # Last chunk, note that it ends at 41.0 but starts at 21.0 to ensure a 20-seconds chunk
        ]

        self.assertEqual(len(processed_chunks), len(expected_chunk_times))

        for (chunk_time_str, tensor), (expected_start, expected_end) in zip(
            processed_chunks, expected_chunk_times
        ):
            # Validate chunk time format and values
            self.assertTrue(
                chunk_time_str.startswith("[") and chunk_time_str.endswith("]")
            )
            start_time, end_time = map(float, chunk_time_str.strip("[]").split(","))

            self.assertAlmostEqual(start_time, expected_start, places=2)
            self.assertAlmostEqual(end_time, expected_end, places=2)

            # Validate tensor correctness
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, torch.Size([1, 10, 10]))

    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor._fetch_file_metadata"
    )
    @patch(
        "marqo.inference.media_download_and_preprocess.streaming_media_processor.StreamingMediaProcessor.fetch_video_chunk"
    )
    def test_explicit_video_chunk_times_without_chunk(
        self, mock_fetch_audio_chunk, mock_fetch_file_metadata
    ):
        mock_fetch_file_metadata.return_value = (1000000, 41.0, Modality.VIDEO)
        mock_fetch_audio_chunk.side_effect = (
            lambda start_time, duration, output_file: output_file
        )

        overlap_config = VideoPreprocessingConfig(
            modality=Modality.VIDEO,
            download_thread_count=1,
            download_header=None,
            should_chunk=False,  # No chunking
        )

        processor = StreamingMediaProcessor(
            url=TestVideoUrls.VIDEO1.value,
            preprocessors=self.test_preprocessor,
            preprocessing_config=overlap_config,
        )

        processed_chunks = processor.process_media()

        # Define the exact expected chunk times
        expected_chunk_times = [
            (0.0, 41.0)  # Only one chunk for the entire video
        ]

        self.assertEqual(len(processed_chunks), len(expected_chunk_times))

        for (chunk_time_str, tensor), (expected_start, expected_end) in zip(
            processed_chunks, expected_chunk_times
        ):
            # Validate chunk time format and values
            self.assertTrue(
                chunk_time_str.startswith("[") and chunk_time_str.endswith("]")
            )
            start_time, end_time = map(float, chunk_time_str.strip("[]").split(","))

            self.assertAlmostEqual(start_time, expected_start, places=2)
            self.assertAlmostEqual(end_time, expected_end, places=2)

            # Validate tensor correctness
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, torch.Size([1, 10, 10]))


class TestChunkTimingGenerator(unittest.TestCase):
    """Unit tests for ChunkTimingGenerator class."""

    def test_basic_functionality_no_overlap(self):
        """Test basic chunking without overlap."""
        generator = ChunkTimingGenerator(
            duration=10.0, chunk_duration=3.0, overlap_duration=0.0
        )
        chunks = list(generator)
        expected = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (7.0, 10.0)]
        self.assertEqual(expected, chunks)

    def test_basic_functionality_with_overlap(self):
        """Test basic chunking with overlap."""
        generator = ChunkTimingGenerator(
            duration=10.0, chunk_duration=3.0, overlap_duration=1.0
        )
        chunks = list(generator)
        expected = [(0.0, 3.0), (2.0, 5.0), (4.0, 7.0), (6.0, 9.0), (7.0, 10.0)]
        self.assertEqual(expected, chunks)

    def test_single_chunk_when_duration_equals_chunk_duration(self):
        """Test single chunk when duration equals chunk duration."""
        generator = ChunkTimingGenerator(
            duration=5.0, chunk_duration=5.0, overlap_duration=1.0
        )
        chunks = list(generator)
        expected = [(0.0, 5.0)]
        self.assertEqual(expected, chunks)

    def test_single_chunk_when_duration_less_than_chunk_duration(self):
        """Test single chunk when duration is less than chunk duration."""
        generator = ChunkTimingGenerator(
            duration=3.0, chunk_duration=5.0, overlap_duration=1.0
        )
        chunks = list(generator)
        expected = [(0.0, 3.0)]
        self.assertEqual(expected, chunks)

    def test_exact_fit_no_overlap(self):
        """Test when duration divides evenly into chunks with no overlap."""
        generator = ChunkTimingGenerator(
            duration=10.0, chunk_duration=5.0, overlap_duration=0.0
        )
        chunks = list(generator)
        expected = [(0.0, 5.0), (5.0, 10.0)]
        self.assertEqual(expected, chunks)

    def test_zero_duration_should_generate_empty_chunk_list(self):
        generator = ChunkTimingGenerator(
            duration=0.0, chunk_duration=5.0, overlap_duration=0.0
        )
        chunks = list(generator)
        self.assertEqual([], chunks)

    def test_overlap_duration_larger_than_duration(self):
        """Test behavior when overlap larger relative to chunk duration."""
        generator = ChunkTimingGenerator(
            duration=5.0, chunk_duration=10.0, overlap_duration=6.0
        )
        chunks = list(generator)
        expected = [(0.0, 5.0)]
        self.assertEqual(expected, chunks)

    def test_overlap_duration_equals_to_duration(self):
        """Test behavior when overlap larger relative to chunk duration."""
        generator = ChunkTimingGenerator(
            duration=5.0, chunk_duration=10.0, overlap_duration=5.0
        )
        chunks = list(generator)
        expected = [(0.0, 5.0)]
        self.assertEqual(expected, chunks)

    def test_negative_duration_should_raise_error(self):
        with self.assertRaises(ValueError) as context:
            ChunkTimingGenerator(
                duration=-1.0, chunk_duration=10.0, overlap_duration=5.0
            )
        self.assertEqual(
            "Duration of the media file is negative: -1.0", str(context.exception)
        )

    def test_negative_step_should_raise_error(self):
        with self.assertRaises(ValueError) as context:
            ChunkTimingGenerator(
                duration=11.0, chunk_duration=4.0, overlap_duration=5.0
            )
        self.assertEqual(
            "Chunking error due to chunk size (4.0) <= overlap (5.0)",
            str(context.exception),
        )

    def test_zero_step_should_raise_error(self):
        with self.assertRaises(ValueError) as context:
            ChunkTimingGenerator(
                duration=11.0, chunk_duration=5.0, overlap_duration=5.0
            )
        self.assertEqual(
            "Chunking error due to chunk size (5.0) <= overlap (5.0)",
            str(context.exception),
        )
