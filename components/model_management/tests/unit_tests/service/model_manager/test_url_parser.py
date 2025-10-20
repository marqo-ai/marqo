from unittest import TestCase

from model_management.services.model_manager.url_parser import get_base_filename


class TestGetBaseFilename(TestCase):
    """Test class for get_base_filename function."""

    def test_get_base_filename_with_http_urls(self):
        """Test get_base_filename with HTTP URLs."""
        test_cases = [
            ("http://example.com/model.onnx", "model.onnx"),
            ("http://example.com/path/to/model.onnx", "model.onnx"),
            ("http://example.com/deep/nested/path/model.onnx.data", "model.onnx.data"),
            ("http://example.com/model.onnx?version=1", "model.onnx"),
            ("http://example.com/model.onnx?key=value&foo=bar", "model.onnx"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_https_urls(self):
        """Test get_base_filename with HTTPS URLs."""
        test_cases = [
            ("https://example.com/model.onnx", "model.onnx"),
            ("https://cdn.example.com/models/v1/model.onnx", "model.onnx"),
            (
                "https://storage.example.com/bucket/model.onnx.data_0",
                "model.onnx.data_0",
            ),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_s3_urls(self):
        """Test get_base_filename with S3 URLs."""
        test_cases = [
            ("s3://bucket/model.onnx", "model.onnx"),
            ("s3://my-bucket/path/to/model.onnx", "model.onnx"),
            ("s3://bucket/models/v2/model.onnx.data", "model.onnx.data"),
            ("s3://bucket/nested/deep/path/model.onnx.data_1", "model.onnx.data_1"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_ftp_urls(self):
        """Test get_base_filename with FTP URLs."""
        test_cases = [
            ("ftp://server.com/model.onnx", "model.onnx"),
            ("ftp://ftp.example.com/public/models/model.onnx", "model.onnx"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_local_paths(self):
        """Test get_base_filename with local file paths."""
        test_cases = [
            ("/absolute/path/to/model.onnx", "model.onnx"),
            ("./relative/path/model.onnx", "model.onnx"),
            ("model.onnx", "model.onnx"),
            ("/tmp/models/model.onnx.data", "model.onnx.data"),
            ("../parent/directory/model.onnx", "model.onnx"),
        ]

        for path, expected in test_cases:
            with self.subTest(path=path):
                self.assertEqual(expected, get_base_filename(path))

    def test_get_base_filename_with_file_scheme(self):
        """Test get_base_filename with file:// URLs."""
        test_cases = [
            ("file:///path/to/model.onnx", "model.onnx"),
            ("file:///tmp/models/model.onnx.data", "model.onnx.data"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_various_extensions(self):
        """Test get_base_filename with various file extensions."""
        test_cases = [
            ("s3://bucket/model.onnx", "model.onnx"),
            ("s3://bucket/model.onnx.data", "model.onnx.data"),
            ("s3://bucket/model.onnx.data_0", "model.onnx.data_0"),
            ("s3://bucket/model.onnx.data_1", "model.onnx.data_1"),
            ("https://example.com/weights.bin", "weights.bin"),
            ("https://example.com/config.json", "config.json"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_preserves_special_characters(self):
        """Test that get_base_filename preserves special characters in filename."""
        test_cases = [
            ("s3://bucket/model-v1.0.onnx", "model-v1.0.onnx"),
            ("https://example.com/model_2024.onnx", "model_2024.onnx"),
            ("s3://bucket/model (1).onnx", "model (1).onnx"),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))

    def test_get_base_filename_with_trailing_slash(self):
        """Test get_base_filename with URLs that have trailing slashes."""
        test_cases = [
            ("s3://bucket/model.onnx/", ""),
            ("https://example.com/path/", ""),
        ]

        for url, expected in test_cases:
            with self.subTest(url=url):
                self.assertEqual(expected, get_base_filename(url))
