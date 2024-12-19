from unittest import TestCase
from unittest.mock import patch, MagicMock, ANY

import pycurl
import pytest
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from marqo.s2_inference.clip_utils import encode_url, download_image_from_url
from marqo.s2_inference.errors import ImageDownloadError
from integ_tests.marqo_test import MockHttpServer
from marqo.tensor_search.enums import EnvVars
from io import BytesIO


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
                    download_image_from_url(image_path=url + ".jpg", media_download_headers={})

    def test_download_image_from_url_handlesUrlRequiringUserAgentHeader(self):
        url_requiring_user_agent_header = "https://docs.marqo.ai/2.0.0/Examples/marqo.jpg"
        try:
            download_image_from_url(image_path=url_requiring_user_agent_header, media_download_headers={})
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
                download_image_from_url('http://example.com/image.jpg', media_download_headers=headers)
                mock_curl_instance.setopt.assert_called_with(pycurl.HTTPHEADER, expected_headers)

    def test_download_image_from_url_handlesRedirection(self):
        image_content = b'\x00\x00\x00\xff'
        app = Starlette(routes=[
            Route('/missing_image.jpg', lambda _: Response(status_code=301, headers={'Location': '/image.jpg'})),
            Route('/image.jpg', lambda _: Response(image_content, media_type='image/png')),
        ])

        with MockHttpServer(app).run_in_thread() as base_url:
            result = download_image_from_url(f'{base_url}/missing_image.jpg', media_download_headers={})
            self.assertEqual(result.getvalue(), image_content)
    
    @patch('marqo.s2_inference.clip_utils.pycurl.Curl')
    @patch.dict('os.environ', {'MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE': '5000000'})  # 5MB limit
    def test_video_audio_file_size_check_over_limit(self, mock_curl):
        # Setup
        test_url = "http://ipv4.download.thinkbroadband.com:8080/5GB.zip"
        mock_curl_instance = MagicMock()
        mock_curl.return_value = mock_curl_instance

        # Store the progress callback
        progress_callback = None

        def simulate_setopt(option, value):
            nonlocal progress_callback
            if option == pycurl.XFERINFOFUNCTION:
                progress_callback = value
            elif option == pycurl.WRITEFUNCTION:
                # Simulate writing some data
                value(b'Some data')

        mock_curl_instance.setopt.side_effect = simulate_setopt
        
        # Simulate pycurl.error with E_ABORTED_BY_CALLBACK
        mock_curl_instance.perform.side_effect = pycurl.error(pycurl.E_ABORTED_BY_CALLBACK, "Callback aborted")

        # Test
        with self.assertRaises(ImageDownloadError) as context:
            download_image_from_url(test_url, {}, modality="video")
            
            # Simulate the progress callback after download_image_from_url has set it
            if progress_callback:
                # Simulate downloading more than the limit
                progress_callback(0, 6_000_000, 0, 0)

        # Assert
        self.assertIn("exceeds the maximum allowed size", str(context.exception))
        mock_curl_instance.setopt.assert_any_call(pycurl.NOPROGRESS, False)
        mock_curl_instance.setopt.assert_any_call(pycurl.XFERINFOFUNCTION, ANY)

    @patch('marqo.s2_inference.clip_utils.pycurl.Curl')
    @patch.dict('os.environ', {'MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE': '5000000'}) # 5MB limit
    def test_video_audio_file_size_check_under_limit(self, mock_curl):
        # Setup
        test_url = "http://example.com/small_video.mp4"
        mock_curl_instance = MagicMock()
        mock_curl.return_value = mock_curl_instance

        # Store the progress callback
        progress_callback = None

        def simulate_setopt(option, value):
            nonlocal progress_callback
            if option == pycurl.XFERINFOFUNCTION:
                progress_callback = value

        mock_curl_instance.setopt.side_effect = simulate_setopt
        mock_curl_instance.getinfo.return_value = 200  # Simulate successful HTTP response

        # Test
        try:
            result = download_image_from_url(test_url, {}, modality="audio")
            
            # Simulate the progress callback after download_image_from_url has set it
            if progress_callback:
                progress_callback(0, 3_000_000, 0, 0)
            
            self.assertIsInstance(result, BytesIO)
        except ImageDownloadError:
            self.fail("ImageDownloadError raised unexpectedly for file under size limit")

        # Assert
        mock_curl_instance.setopt.assert_any_call(pycurl.NOPROGRESS, False)
        mock_curl_instance.setopt.assert_any_call(pycurl.XFERINFOFUNCTION, ANY)
        mock_curl_instance.perform.assert_called_once()

    @patch('marqo.s2_inference.clip_utils.pycurl.Curl')
    @patch('marqo.s2_inference.clip_utils.EnvVars.MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE', 5_000_000)  # 5MB limit
    def test_image_file_size_not_checked(self, mock_curl):
        # Setup
        test_url = "http://example.com/large_image.jpg"
        mock_curl_instance = MagicMock()
        mock_curl.return_value = mock_curl_instance

        # Simulate successful download
        mock_curl_instance.getinfo.return_value = 200

        # Test
        try:
            result = download_image_from_url(test_url, {}, modality="image")
            self.assertIsInstance(result, BytesIO)
        except ImageDownloadError:
            self.fail("ImageDownloadError raised unexpectedly for image modality")

        # Assert
        mock_curl_instance.setopt.assert_any_call(pycurl.URL, test_url)
        mock_curl_instance.setopt.assert_any_call(pycurl.WRITEDATA, ANY)
        
        # Check that NOPROGRESS and XFERINFOFUNCTION are not set for image modality
        with self.assertRaises(AssertionError):
            mock_curl_instance.setopt.assert_any_call(pycurl.NOPROGRESS, False)
        with self.assertRaises(AssertionError):
            mock_curl_instance.setopt.assert_any_call(pycurl.XFERINFOFUNCTION, ANY)

        # Verify that perform was called
        mock_curl_instance.perform.assert_called_once()