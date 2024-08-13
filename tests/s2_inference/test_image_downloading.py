from unittest import TestCase
from unittest.mock import patch, MagicMock

import pycurl
import pytest

from marqo.s2_inference.clip_utils import encode_url, download_image_from_url
from marqo.s2_inference.errors import ImageDownloadError


@pytest.mark.unittest
class TestImageDownloading(TestCase):

    def setUp(self):
        self.test_cases = [
            ("http://example.com", "http://example.com", "Basic URL"),
            ("http://example.com/test url", "http://example.com/test%20url", "URL with spaces"),
            ("http://example.com/test!@$&*()_+={}[]|\\:;'\"<>,.?/",
             "http://example.com/test!@$&*()_+=%7B%7D[]%7C%5C:;'%22%3C%3E,.?/", "URL with special characters"),
            ("http://example.com/你好世界", "http://example.com/%E4%BD%A0%E5%A5%BD%E4%B8%96%E7%95%8C",
             "URL with non-ASCII characters"),
            ("http://example.com/test?name=John Doe&age=30", "http://example.com/test?name=John%20Doe&age=30",
             "URL with query parameters"),
            ("http://example.com/test#section 1", "http://example.com/test#section%201", "URL with fragments"),
            ("http://example.com//test//path", "http://example.com//test//path", "URL with multiple slashes"),
            ("http://example.com/test%20url", "http://example.com/test%20url", "URL with encoded characters"),
            ("http://example.com/test url%20example", "http://example.com/test%20url%20example",
             "URL with mixed encoded and unencoded characters"),
            ("http://example.com/例子.测试.jpg", "http://example.com/%E4%BE%8B%E5%AD%90.%E6%B5%8B%E8%AF%95.jpg",
             "URL with unicode characters in the domain"),
            ("http://example.com/" + "a" * 2000, "http://example.com/" + "a" * 2000, "Long URL"),
            ("https://example.com", "https://example.com", "URL with HTTPS scheme"),
            ("ftp://example.com", "ftp://example.com", "URL with FTP scheme"),
            ("", "", "Empty URL"),
            ("http://example.com/œ∑ł.jpg", "http://example.com/%C5%93%E2%88%91%C5%82.jpg",
             "URL with unicode characters in the path"),
            ("http://127.0.0.1/test", "http://127.0.0.1/test", "URL with IP address"),
        ]

    def test_encode_url_handleDifferentUrlsCorrectly(self):
        for url, expected, msg in self.test_cases:
            with self.subTest(url=url, expected=expected, msg=msg):
                result = encode_url(url)
                self.assertEqual(result, expected, f"Error: for {msg}, expected '{expected}', but got '{result}'")
                self.assertEqual(encode_url(url), expected)

    def test_download_image_from_url_handleDifferentUrlsCorrectly(self):
        """Ensure no 500 error is raised when downloading images from different URLs."""
        for url, expected, msg in self.test_cases:
            with self.subTest(url=url, expected=expected, msg=msg):
                with self.assertRaises(ImageDownloadError) as cm:
                    download_image_from_url(image_path=url + ".jpg", image_download_headers={})

    def test_download_image_from_url_handlesUrlRequiringUserAgentHeader(self):
        url_requiring_user_agent_header = "https://docs.marqo.ai/2.0.0/Examples/marqo.jpg"
        try:
            download_image_from_url(image_path=url_requiring_user_agent_header, image_download_headers={})
        except Exception as e:
            self.fail(f"Exception was raised when downloading {url_requiring_user_agent_header}: {e}")

    @patch('pycurl.Curl')
    def test_download_image_from_url_mergesDefaultHeadersWithCustomHeaders(self, mock_curl):
        mock_curl_instance = mock_curl.return_value
        mock_curl_instance.setopt = MagicMock()
        mock_curl_instance.perform = MagicMock()
        mock_curl_instance.getinfo = MagicMock(return_value=200)

        test_cases = [
            ({}, ['User-Agent: Marqobot/1.0'], "Empty header"),
            ({'a': 'b'}, ['User-Agent: Marqobot/1.0', 'a: b'], "Basic headers"),
            ({'User-Agent': 'Marqobot-Image/1.0'}, ['User-Agent: Marqobot-Image/1.0'], "Headers with override"),
        ]

        for (headers, expected_headers, msg) in test_cases:
            with self.subTest(headers=headers, expected_headers=expected_headers, msg=msg):
                download_image_from_url('http://example.com/image.jpg', image_download_headers=headers)
                mock_curl_instance.setopt.assert_called_with(pycurl.HTTPHEADER, expected_headers)