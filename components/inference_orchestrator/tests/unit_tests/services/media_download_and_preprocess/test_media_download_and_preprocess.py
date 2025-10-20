from unittest import TestCase
from unittest.mock import MagicMock, patch

from inference_orchestrator.api.telemetry import RequestMetrics
from inference_orchestrator.schemas.api import (
    ImagePreprocessingConfig,
    InferenceErrorModel,
    Modality,
)
from inference_orchestrator.services.errors import MediaDownloadError
from inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess import (
    process_batch,
    reduce_thread_metrics,
    threaded_download_and_preprocess_content,
)


class TestMediaDownloadAndPreprocess(TestCase):
    def setUp(self):
        self.sample_image_preprocessing_config = ImagePreprocessingConfig(
            modality=Modality.IMAGE,
            download_header={"Authorization": "Bearer fake_token"},
            download_timeout_ms=1000,
            download_thread_count=2,
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
                return_individual_error=True,
            )
        self.assertIn("Unsupported modality", str(context.exception))

    def test_reduce_thread_metrics(self):
        raw_data = {
            "media_download.image.100.thread_time": 10,
            "media_download.image.100.url1": 20,
            "media_download.image.101.thread_time": 15,
            "media_download.image.101.url1": 25,
        }

        reduced = reduce_thread_metrics(raw_data)

        self.assertIn("media_download.image.thread_time", reduced)
        self.assertEqual([10, 15], reduced["media_download.image.thread_time"])
        self.assertEqual(
            [20, 25],
            reduced["media_download.image.url1"],
        )

    def test_process_batch_raises_on_thread_error(self):
        content = ["url1", "url2"]

        # Simulate one thread raising an error
        def mock_threaded_download_and_preprocess_content(*args, **kwargs):
            allocated_content = args[0]
            if "url1" in allocated_content:
                raise MediaDownloadError("Simulated thread error")
            return [[(url, "tensor")] for url in allocated_content]

        with patch(
            "inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess."
            "threaded_download_and_preprocess_content",
            side_effect=mock_threaded_download_and_preprocess_content,
        ):
            with self.assertRaises(MediaDownloadError) as context:
                process_batch(
                    content=content,
                    preprocessor=MagicMock(),
                    preprocessing_config=self.sample_image_preprocessing_config,
                    return_individual_error=False,
                )
            self.assertIn("Simulated thread error", str(context.exception))

    def test_process_batch_collects_errors_with_return_individual_error_true(self):
        content = ["url1", "url2"]

        # Simulate one thread raising an error, other processes fine
        def mock_threaded_download_and_preprocess_content(*args, **kwargs):
            allocated_content = args[0]
            if "url1" in allocated_content:
                return [InferenceErrorModel(error_message="Simulated error for url1")]
            return [[(url, "tensor")] for url in allocated_content]

        with (
            patch(
                "inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess."
                "threaded_download_and_preprocess_content",
                side_effect=mock_threaded_download_and_preprocess_content,
            ),
            patch(
                "inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess."
                "RequestMetricsStore.for_request",
                return_value=RequestMetrics(),
            ),
        ):
            results = process_batch(
                content=content,
                preprocessor=MagicMock(),
                preprocessing_config=self.sample_image_preprocessing_config,
                return_individual_error=True,
            )

            # Expect results for both URLs
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], InferenceErrorModel)
            self.assertIn("Simulated error for url1", results[0].error_message)
            self.assertIsInstance(results[1], list)
            self.assertEqual(results[1][0][0], "url2")
