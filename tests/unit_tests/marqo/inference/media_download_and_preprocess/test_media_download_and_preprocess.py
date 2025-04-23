from unittest import TestCase
from unittest.mock import patch, MagicMock

from marqo.core.inference.api import *
from marqo.inference.media_download_and_preprocess.media_download_and_preprocess import (
    _threaded_download_and_preprocess_audio_and_video,
    threaded_download_and_preprocess_content,
    reduce_thread_metrics,
    _enable_video_gpu_acceleration,
    process_batch
)
from marqo.tensor_search.telemetry import RequestMetrics


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
                preprocessin_config=config,
                metric_obj=None,
                return_individual_error=True
            )
        self.assertIn("Unsupported modality", str(context.exception))

    def test_reduce_thread_metrics(self):
        raw_data = {
            "image_download.100.thread_time": 10,
            "image_download.100.url1": 20,
            "image_download.101.thread_time": 15,
            "image_download.101.url1": 25
        }

        reduced = reduce_thread_metrics(raw_data)

        self.assertIn("image_download.thread_time", reduced)
        self.assertEqual(reduced["image_download.thread_time"], [10, 15])
        self.assertEqual(reduced["image_download.url1"], [20, 25])

    def test_enable_video_gpu_acceleration_true(self):
        with patch("marqo.tensor_search.utils.read_env_vars_and_defaults", return_value="TRUE"):
            self.assertTrue(_enable_video_gpu_acceleration())

    def test_threaded_download_and_preprocess_audio_success(self):
        fake_preprocessor = MagicMock()
        fake_results = [("chunk1", "tensor1")]

        with patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.StreamingMediaProcessor") as mock_processor:
            instance = mock_processor.return_value
            instance.process_media.return_value = fake_results

            result = _threaded_download_and_preprocess_audio_and_video(
                allocated_content=["audio_url1"],
                preprocessor=fake_preprocessor,
                preprocessing_config=self.sample_audio_preprocessing_config,
                metric_obj=RequestMetrics(),
                return_individual_error=True
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], fake_results)

    def test_threaded_download_and_preprocess_video_success(self):
        fake_preprocessor = MagicMock()
        fake_results = [("chunk1", "tensor1")]

        with patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.StreamingMediaProcessor") as mock_processor:
            instance = mock_processor.return_value
            instance.process_media.return_value = fake_results

            result = _threaded_download_and_preprocess_audio_and_video(
                allocated_content=["video_url1"],
                preprocessor=fake_preprocessor,
                preprocessing_config=self.sample_video_preprocessing_config,
                metric_obj=RequestMetrics(),
                return_individual_error=True
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], fake_results)

    def test_threaded_download_and_preprocess_audio_error(self):
        fake_preprocessor = MagicMock()

        with patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.StreamingMediaProcessor") as mock_processor:
            instance = mock_processor.return_value
            instance.process_media.side_effect = InferenceError("Download failed")

            result = _threaded_download_and_preprocess_audio_and_video(
                allocated_content=["audio_url1"],
                preprocessor=fake_preprocessor,
                preprocessing_config=self.sample_audio_preprocessing_config,
                metric_obj=RequestMetrics(),
                return_individual_error=True
            )

            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], InferenceErrorModel)
            self.assertIn("Download failed", result[0].error_message)

    def test_process_batch_raises_on_thread_error(self):
        content = ["url1", "url2"]

        # Simulate one thread raising an error
        def mock_threaded_download_and_preprocess_content(*args, **kwargs):
            allocated_content = args[0]
            if "url1" in allocated_content:
                raise MediaDownloadError("Simulated thread error")
            return [[(url, "tensor")] for url in allocated_content]

        with patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.threaded_download_and_preprocess_content", side_effect=mock_threaded_download_and_preprocess_content):
            with self.assertRaises(MediaDownloadError) as context:
                process_batch(
                    content=content,
                    preprocessor=MagicMock(),
                    preprocessing_config=self.sample_image_preprocessing_config,
                    return_individual_error=False
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

        with patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.threaded_download_and_preprocess_content", side_effect=mock_threaded_download_and_preprocess_content), \
             patch("marqo.inference.media_download_and_preprocess.media_download_and_preprocess.RequestMetricsStore.for_request", return_value=RequestMetrics()):
            results = process_batch(
                content=content,
                preprocessor=MagicMock(),
                preprocessing_config=self.sample_image_preprocessing_config,
                return_individual_error=True
            )

            # Expect results for both URLs
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], InferenceErrorModel)
            self.assertIn("Simulated error for url1", results[0].error_message)
            self.assertIsInstance(results[1], list)
            self.assertEqual(results[1][0][0], "url2")