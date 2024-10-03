import os.path
import unittest

from pytest import mark

from marqo.s2_inference.errors import MediaDownloadError
from marqo.tensor_search.streaming_media_processor import StreamingMediaProcessor


class TestVideoFFmpegDecode(unittest.TestCase):
    def setUp(self):
        self.output_file = "./test.mp4"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_video_decode_cpu_works(self):
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_gpu_acceleration = False

        StreamingMediaProcessor.fetch_video_chunk(
            valid_url, start_time, duration, self.output_file, enable_gpu_acceleration
        )
        self.assertTrue(os.path.exists(self.output_file))

    def test_video_decode_cpu_invalid_url(self):
        invalid_url = "https://rqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_gpu_acceleration = False

        with self.assertRaises(MediaDownloadError) as e:
            StreamingMediaProcessor.fetch_video_chunk(
                invalid_url, start_time, duration, self.output_file, enable_gpu_acceleration
            )
        self.assertFalse(os.path.exists(self.output_file))
        self.assertIn("404", str(e.exception))

    @mark.cpu_only
    def test_video_decode_gpu_does_not_work(self):
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1
        enable_gpu_acceleration = True

        with self.assertRaises(MediaDownloadError):
            StreamingMediaProcessor.fetch_video_chunk(
                valid_url, start_time, duration, self.output_file, enable_gpu_acceleration
            )
        self.assertFalse(os.path.exists(self.output_file))

    @mark.largemodel
    def test_video_decode_cuda_works(self):
        valid_url = "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/--_S9IDQPLg_000135_000145.mp4"
        start_time = 0
        duration = 1

        for enable_gpu_acceleration in (True, False):
            StreamingMediaProcessor.fetch_video_chunk(
                valid_url, start_time, duration, self.output_file, enable_gpu_acceleration
            )
            self.assertTrue(os.path.exists(self.output_file))
            os.remove(self.output_file)