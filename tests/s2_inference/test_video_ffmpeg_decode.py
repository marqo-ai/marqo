import os.path
import unittest

from pytest import mark

from marqo.s2_inference.errors import MediaDownloadError
from marqo.tensor_search.streaming_media_processor import StreamingMediaProcessor
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.s2_inference.multimodal_model_load import Modality


class TestVideoFFmpegDecode(unittest.TestCase):
    def setUp(self):
        self.output_file = "./test.mp4"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    @mark.cpu_only
    def test_video_decode_cpu_works(self):
        """Video decoding should work on a CPU-only machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_video_gpu_acceleration = False

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, device="cpu", modality=Modality.VIDEO, preprocessors=Preprocessors(),
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
                url=invalid_url, device="cpu", modality=Modality.VIDEO, preprocessors=Preprocessors(),
                enable_video_gpu_acceleration=enable_video_gpu_acceleration
            )

        self.assertFalse(os.path.exists(self.output_file))
        self.assertIn("404", str(e.exception))

    @mark.cpu_only
    def test_video_decode_gpu_does_not_work(self):
        """A proper error is raised when trying to decode a video with GPU acceleration enabled on a CPU-only machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_video_gpu_acceleration = True

        streaming_media_processor_object = StreamingMediaProcessor(
            url=valid_url, device="cpu", modality=Modality.VIDEO, preprocessors=Preprocessors(),
            enable_video_gpu_acceleration=enable_video_gpu_acceleration
        )

        with self.assertRaises(MediaDownloadError) as e:
            streaming_media_processor_object.fetch_video_chunk(start_time, duration, self.output_file)
        self.assertFalse(os.path.exists(self.output_file))

    @mark.largemodel
    def test_video_decode_cuda_works(self):
        """Both CPU and GPU decoding should work on a GPU-enabled machine."""
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1

        for enable_video_gpu_acceleration in (True, False):
            streaming_media_processor_object = StreamingMediaProcessor(
                url=valid_url, device="cpu", modality=Modality.VIDEO, preprocessors=Preprocessors(),
                enable_video_gpu_acceleration=enable_video_gpu_acceleration
            )
            streaming_media_processor_object.fetch_video_chunk(
                start_time, duration, self.output_file
            )
            self.assertTrue(os.path.exists(self.output_file))
            os.remove(self.output_file)