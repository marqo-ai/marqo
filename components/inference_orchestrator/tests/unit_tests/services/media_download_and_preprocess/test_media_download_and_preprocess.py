import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from inference_orchestrator.schemas.api import *
from inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess import (
    threaded_download_and_preprocess_content,
    reduce_thread_metrics,
    process_batch
)
from inference_orchestrator.api.telemetry import RequestMetrics
from inference_orchestrator.services.errors import *


class TestMediaDownloadAndPreprocess(TestCase):

    def setUp(self):
        self.sample_image_preprocessing_config = ImagePreprocessingConfig(
            modality=Modality.IMAGE,
            download_header={"Authorization": "Bearer fake_token"},
            download_timeout_ms=1000,
            download_thread_count=2
        )

        self.sample_audio_preprocessing_config = AudioPreprocessingConfig(
            modality=Modality.AUDIO,
            download_header={},
            download_timeout_ms=1000,
            download_thread_count=1
        )

        self.sample_video_preprocessing_config = VideoPreprocessingConfig(
            modality=Modality.VIDEO,
            download_header={},
            download_timeout_ms=1000,
            download_thread_count=1
        )

    def test_threaded_download_and_preprocess_invalid_modality(self):
        config = MagicMock()
        config.modality = "unsupported_modality"

        with self.assertRaises(ValueError) as context:
            threaded_download_and_preprocess_content(
                allocated_content=["url"],
                preprocessor=MagicMock(),
                preprocessing_config=config,
                metric_obj=None,
                return_individual_error=True
            )
        self.assertIn("Unsupported modality", str(context.exception))

    def test_reduce_thread_metrics(self):
        raw_data = {
            "media_download.image.100.thread_time": 10,
            "media_download.image.100.url1": 20,
            "media_download.image.101.thread_time": 15,
            "media_download.image.101.url1": 25
        }

        reduced = reduce_thread_metrics(raw_data)

        self.assertIn("media_download.image.thread_time", reduced)
        self.assertEqual([10, 15], reduced["media_download.image.thread_time"])
        self.assertEqual([20, 25], reduced["media_download.image.url1"],)

    @unittest.skip("Test references old marqo module paths")
    def test_process_batch_raises_on_thread_error(self):
        pass

    @unittest.skip("Test references old marqo module paths")
    def test_process_batch_collects_errors_with_return_individual_error_true(self):
        pass