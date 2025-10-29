import unittest

from inference_orchestrator.services.triton_inference.embedding_models.url_parser import (
    get_base_filename,
)


class TestGetBaseFilename(unittest.TestCase):
    """Tests for get_base_filename function."""

    def test_get_base_filename_from_http_url(self):
        """Test extracting filename from HTTP URL."""
        url = "http://example.com/path/to/file.txt"
        result = get_base_filename(url)
        self.assertEqual("file.txt", result)

    def test_get_base_filename_from_https_url(self):
        """Test extracting filename from HTTPS URL."""
        url = "https://example.com/path/to/model.bin"
        result = get_base_filename(url)
        self.assertEqual("model.bin", result)

    def test_get_base_filename_from_ftp_url(self):
        """Test extracting filename from FTP URL."""
        url = "ftp://server.com/files/data.csv"
        result = get_base_filename(url)
        self.assertEqual("data.csv", result)

    def test_get_base_filename_from_s3_url(self):
        """Test extracting filename from S3 URL."""
        url = "s3://bucket/path/to/object.json"
        result = get_base_filename(url)
        self.assertEqual("object.json", result)

    def test_get_base_filename_from_url_with_query_params(self):
        """Test that query parameters are ignored."""
        url = "https://example.com/path/file.txt?version=1&download=true"
        result = get_base_filename(url)
        self.assertEqual("file.txt", result)

    def test_get_base_filename_from_local_path(self):
        """Test extracting filename from local file path."""
        path = "/home/user/documents/report.pdf"
        result = get_base_filename(path)
        self.assertEqual("report.pdf", result)

    def test_get_base_filename_from_relative_path(self):
        """Test extracting filename from relative path."""
        path = "folder/subfolder/file.txt"
        result = get_base_filename(path)
        self.assertEqual("file.txt", result)

    def test_get_base_filename_from_windows_path(self):
        """Test extracting filename from Windows-style path."""
        import os

        # Use os.path.join to create a platform-appropriate path
        path = os.path.join("C:", "Users", "Name", "file.txt")
        result = get_base_filename(path)
        self.assertEqual("file.txt", result)

    def test_get_base_filename_single_filename(self):
        """Test with just a filename (no path)."""
        filename = "standalone.txt"
        result = get_base_filename(filename)
        self.assertEqual("standalone.txt", result)


if __name__ == "__main__":
    unittest.main()
