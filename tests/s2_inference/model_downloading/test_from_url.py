import os
import unittest
from unittest.mock import patch, Mock, MagicMock
import hashlib
from typing import Optional, Union
from marqo.s2_inference.processing.custom_clip_utils import download_pretrained_from_url, calculate_sha256_checksum

class TestModelDownload(unittest.TestCase):

    def setUp(self):
        self.test_url = "https://example.com/model.pt"
        self.test_cache_dir = "/tmp/model_cache"
        self.test_cache_file_name = "model_test.pt"
        self.dummy_data = b"dummy model data"
        self.dummy_checksum = hashlib.sha256(self.dummy_data).hexdigest()

    @patch("os.path.isfile")
    @patch("os.remove")
    @patch("urllib.request.urlopen")
    @patch("builtins.open")
    def test_download_pretrained_from_url_no_existing_file(self, mock_open, mock_urlopen, mock_remove, mock_isfile):
        mock_isfile.return_value = False

        mock_read = MagicMock()
        mock_read.read.side_effect = [self.dummy_data, b""]
        mock_urlopen.return_value.__enter__.return_value = mock_read

        # Call the function
        download_pretrained_from_url(self.test_url, self.test_cache_dir, self.test_cache_file_name, self.dummy_checksum)

        # Assert the function calls
        mock_open.assert_called_with(os.path.join(self.test_cache_dir, self.test_cache_file_name), "wb")
        mock_urlopen.assert_called_with(self.test_url)

    @patch("builtins.open")
    def test_calculate_sha256_checksum(self, mock_open):
        mock_file = MagicMock()
        mock_file.read.side_effect = [self.dummy_data, b""]
        mock_open.return_value.__enter__.return_value = mock_file

        # Call the function
        actual_checksum = calculate_sha256_checksum("dummy_path")

        # Assert the function calls and result
        mock_open.assert_called_with("dummy_path", "rb")
        self.assertEqual(self.dummy_checksum, actual_checksum)

    @patch("os.path.isfile")
    @patch("os.remove")
    @patch("urllib.request.urlopen")
    @patch("builtins.open")
    def test_download_pretrained_from_url_existing_file_different_checksum(self, mock_open, mock_urlopen, mock_remove, mock_isfile):
        mock_isfile.return_value = True

        # Mock the calculate_sha256_checksum function to return a different checksum
        with patch('calculate_sha256_checksum', return_value="different_checksum"):
            mock_read = MagicMock()
            mock_read.read.side_effect = [self.dummy_data, b""]
            mock_urlopen.return_value.__enter__.return_value = mock_read

            # Call the function
            download_pretrained_from_url(self.test_url, self.test_cache_dir, self.test_cache_file_name, self.dummy_checksum)

            # Assert the function calls
            expected_path = os.path.join(self.test_cache_dir, self.test_cache_file_name)
            mock_remove.assert_called_with(expected_path)
            mock_open.assert_called_with(expected_path, "wb")
            mock_urlopen.assert_called_with(self.test_url)
