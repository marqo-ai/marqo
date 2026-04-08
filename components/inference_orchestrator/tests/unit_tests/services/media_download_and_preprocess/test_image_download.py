import base64
import socket
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import pycurl
from PIL import Image, UnidentifiedImageError

from inference_orchestrator.services.errors import (
    ImageDownloadError,
    InternalServerError,
)
from inference_orchestrator.services.media_download_and_preprocess.image_download import (
    _AAAA_CACHE,
    _do_download,
    _get_proxy_headers,
    _is_proxy_failure,
    _load_base64_image,
    _maybe_proxy_url,
    _origin_has_ipv6,
    download_image_from_url,
    download_media_from_url,
    encode_url,
    get_allowed_image_types,
    is_base64_image,
    load_image_from_path,
)


class TestGetAllowedImageTypes(unittest.TestCase):
    """Tests for get_allowed_image_types function."""

    def test_get_allowed_image_types_returns_set(self):
        """Test that get_allowed_image_types returns a set of allowed types."""
        result = get_allowed_image_types()
        self.assertIsInstance(result, set)
        self.assertEqual({".jpg", ".png", ".bmp", ".jpeg"}, result)


class TestIsBase64Image(unittest.TestCase):
    """Tests for is_base64_image function."""

    def test_is_base64_image_valid_data_url(self):
        """Test that is_base64_image correctly identifies data URLs."""
        test_cases = [
            ("png", "data:image/png;base64,iVBORw0KGgoAAAANS"),
            ("jpeg", "data:image/jpeg;base64,/9j/4AAQSkZJRgABA"),
            ("gif", "data:image/gif;base64,R0lGODlhAQABAAAA"),
        ]
        for msg, data_url in test_cases:
            with self.subTest(msg=msg):
                self.assertTrue(is_base64_image(data_url))

    def test_is_base64_image_invalid_inputs(self):
        """Test that is_base64_image returns False for non-base64 inputs."""
        test_cases = [
            ("regular_string", "regular string"),
            ("url", "http://example.com/image.png"),
            ("file_path", "/path/to/image.png"),
            ("empty", ""),
            ("number", 123),
            ("none", None),
        ]
        for msg, input_val in test_cases:
            with self.subTest(msg=msg):
                self.assertFalse(is_base64_image(input_val))


class TestLoadBase64Image(unittest.TestCase):
    """Tests for _load_base64_image function."""

    def test_load_base64_image_valid(self):
        """Test loading a valid base64 image."""
        test_image = Image.new("RGB", (5, 5), color="blue")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        test_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{test_base64}"

        result = _load_base64_image(data_url)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual((5, 5), result.size)

    def test_load_base64_image_without_prefix(self):
        """Test loading base64 image without data URL prefix."""
        test_image = Image.new("RGB", (3, 3), color="green")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        test_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"base64,{test_base64}"

        result = _load_base64_image(data_url)

        self.assertIsInstance(result, Image.Image)

    def test_load_base64_image_invalid_raises_error(self):
        """Test that invalid base64 data raises UnidentifiedImageError."""
        with self.assertRaises(UnidentifiedImageError) as context:
            _load_base64_image("data:image/png;base64,invalid!!!")
        self.assertIn("Invalid base64", str(context.exception))


class TestLoadImageFromPath(unittest.TestCase):
    """Tests for load_image_from_path function."""

    def test_load_image_from_path_with_metrics(self):
        """Test loading image from URL with metrics tracking."""
        with patch("validators.url") as mock_validator:
            with patch(
                "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
            ) as mock_download:
                with patch("PIL.Image.open") as mock_pil_open:
                    with patch("os.path.isfile") as mock_isfile:
                        mock_isfile.return_value = False
                        mock_validator.return_value = True
                        mock_buffer = BytesIO()
                        mock_download.return_value = mock_buffer
                        mock_image = MagicMock()
                        mock_pil_open.return_value = mock_image

                        # Create mock metrics object
                        mock_metrics = MagicMock()

                        result = load_image_from_path(
                            "https://example.com/image.png",
                            {"header": "value"},
                            3000,
                            mock_metrics,
                        )

                        # Verify metrics methods were called
                        mock_metrics.start.assert_called_once_with(
                            "media_download.image.https://example.com/image.png"
                        )
                        mock_metrics.stop.assert_called_once_with(
                            "media_download.image.https://example.com/image.png"
                        )
                        self.assertEqual(mock_image, result)

    def test_load_image_from_path_with_base64_data_url(self):
        """Test loading base64 image from data URL format."""
        test_image = Image.new("RGB", (2, 2), color="red")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        test_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        test_data_url = f"data:image/png;base64,{test_base64}"

        img = load_image_from_path(test_data_url, {})
        self.assertIsInstance(img, Image.Image)
        self.assertEqual((2, 2), img.size)

    def test_load_image_from_path_with_invalid_base64(self):
        """Test error handling for invalid base64 data."""
        invalid_cases = [
            "invalid_base64!!!",
            "data:image/png;base64,invalid!!!",
            "data:image/xxxyyyzzz",
        ]

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(UnidentifiedImageError):
                    load_image_from_path(case, {})

    @patch("os.path.isfile")
    @patch("PIL.Image.open")
    def test_load_image_from_path_local_file(self, mock_open, mock_isfile):
        """Test loading image from local file path."""
        mock_isfile.return_value = True
        mock_image = MagicMock()
        mock_open.return_value = mock_image

        result = load_image_from_path("/path/to/image.png", {})

        mock_isfile.assert_called_with("/path/to/image.png")
        mock_open.assert_called_with("/path/to/image.png")
        self.assertEqual(mock_image, result)

    @patch("validators.url")
    @patch(
        "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
    )
    @patch("PIL.Image.open")
    @patch("os.path.isfile")
    def test_load_image_from_path_url(
        self, mock_isfile, mock_pil_open, mock_download, mock_validator
    ):
        """Test loading image from URL."""
        mock_isfile.return_value = False
        mock_validator.return_value = True
        mock_buffer = BytesIO()
        mock_download.return_value = mock_buffer
        mock_image = MagicMock()
        mock_pil_open.return_value = mock_image

        result = load_image_from_path(
            "https://example.com/image.png", {"header": "value"}
        )

        mock_download.assert_called_once_with(
            "https://example.com/image.png", {"header": "value"}, 3000
        )
        mock_pil_open.assert_called_with(mock_buffer)
        self.assertEqual(mock_image, result)

    @patch("validators.url")
    @patch("os.path.isfile")
    def test_load_image_from_path_invalid_path_raises_error(
        self, mock_isfile, mock_validator
    ):
        """Test that invalid path raises UnidentifiedImageError."""
        mock_isfile.return_value = False
        mock_validator.return_value = False

        with self.assertRaises(UnidentifiedImageError) as context:
            load_image_from_path("invalid_path", {})
        self.assertIn("not a local file", str(context.exception))

    @patch("validators.url")
    @patch(
        "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
    )
    @patch("os.path.isfile")
    def test_load_image_from_path_download_error(
        self, mock_isfile, mock_download, mock_validator
    ):
        """Test that ImageDownloadError is converted to UnidentifiedImageError."""
        mock_isfile.return_value = False
        mock_validator.return_value = True
        mock_download.side_effect = ImageDownloadError("Download failed")

        with self.assertRaises(UnidentifiedImageError) as context:
            load_image_from_path("https://example.com/image.png", {})
        self.assertIn("Download failed", str(context.exception))

    @patch("validators.url")
    @patch(
        "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
    )
    @patch("PIL.Image.open")
    @patch("os.path.isfile")
    def test_load_image_from_path_decoder_error(
        self, mock_isfile, mock_pil_open, mock_download, mock_validator
    ):
        """Test that decoder errors are handled properly."""
        mock_isfile.return_value = False
        mock_validator.return_value = True
        mock_download.return_value = BytesIO()
        mock_pil_open.side_effect = OSError("could not create decoder object")

        with self.assertRaises(UnidentifiedImageError) as context:
            load_image_from_path("https://example.com/image.png", {})
        self.assertIn("could not be decoded", str(context.exception))

    @patch("validators.url")
    @patch(
        "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
    )
    @patch("PIL.Image.open")
    @patch("os.path.isfile")
    def test_load_image_from_path_other_os_error(
        self, mock_isfile, mock_pil_open, mock_download, mock_validator
    ):
        """Test that other OS errors are re-raised."""
        mock_isfile.return_value = False
        mock_validator.return_value = True
        mock_download.return_value = BytesIO()
        mock_pil_open.side_effect = OSError("some other error")

        with self.assertRaises(OSError) as context:
            load_image_from_path("https://example.com/image.png", {})
        self.assertIn("some other error", str(context.exception))


class TestDownloadImageFromUrl(unittest.TestCase):
    """Tests for download_image_from_url function."""

    @patch("pycurl.Curl")
    def test_download_image_from_url_success(self, mock_curl_class):
        """Test successful image download."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        result = download_image_from_url("https://example.com/image.png", {}, 5000)

        self.assertIsInstance(result, BytesIO)
        mock_curl.setopt.assert_any_call(pycurl.TIMEOUT_MS, 5000)
        mock_curl.perform.assert_called_once()
        mock_curl.close.assert_called_once()

    @patch("pycurl.Curl")
    def test_download_image_from_url_non_200_status(self, mock_curl_class):
        """Test that non-200 status codes raise ImageDownloadError."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 404

        with self.assertRaises(ImageDownloadError) as context:
            download_image_from_url("https://example.com/image.png", {}, 3000)
        self.assertIn("404", str(context.exception))
        mock_curl.close.assert_called_once()

    def test_download_image_from_url_invalid_timeout(self):
        """Test that non-integer timeout raises InternalServerError."""
        with self.assertRaises(InternalServerError) as context:
            download_image_from_url("https://example.com/image.png", {}, "not_int")
        self.assertIn("timeout must be an integer", str(context.exception))

    @patch("pycurl.Curl")
    def test_download_image_from_url_pycurl_error(self, mock_curl_class):
        """Test that pycurl errors are converted to ImageDownloadError."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.perform.side_effect = pycurl.error(6, "Could not resolve host")

        with self.assertRaises(ImageDownloadError) as context:
            download_image_from_url("https://example.com/image.png", {}, 3000)
        self.assertIn("Could not resolve host", str(context.exception))
        mock_curl.close.assert_called_once()

    @patch("pycurl.Curl")
    def test_download_image_from_url_size_limit_exceeded(self, mock_curl_class):
        """Test that E_ABORTED_BY_CALLBACK error provides appropriate message for size limit."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.perform.side_effect = pycurl.error(
            pycurl.E_ABORTED_BY_CALLBACK, "Callback aborted"
        )

        with self.assertRaises(ImageDownloadError) as context:
            download_image_from_url("https://example.com/video.mp4", {}, 3000, "video")
        self.assertIn("exceeds the maximum allowed size", str(context.exception))
        self.assertIn("video", str(context.exception))
        mock_curl.close.assert_called_once()

    @patch("pycurl.Curl")
    def test_download_image_from_url_with_headers(self, mock_curl_class):
        """Test that custom headers are included in request."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        custom_headers = {"Authorization": "Bearer token"}
        download_image_from_url("https://example.com/image.png", custom_headers, 3000)

        # Verify headers were set (should include both default and custom headers)
        calls = mock_curl.setopt.call_args_list
        header_call = [call for call in calls if call[0][0] == pycurl.HTTPHEADER]
        self.assertEqual(1, len(header_call))

    def test_download_image_from_url_unicode_encode_error(self):
        """Test that UnicodeEncodeError during URL encoding is handled."""
        with patch(
            "inference_orchestrator.services.media_download_and_preprocess.image_download.encode_url"
        ) as mock_encode:
            mock_encode.side_effect = UnicodeEncodeError("utf-8", "", 0, 1, "invalid")

            with self.assertRaises(ImageDownloadError) as context:
                download_image_from_url("https://example.com/image.png", {}, 3000)
            self.assertIn("could not be encoded", str(context.exception))

    @patch("pycurl.Curl")
    def test_download_image_from_url_with_none_headers(self, mock_curl_class):
        """Test that None headers are handled properly."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        download_image_from_url("https://example.com/image.png", None, 3000)

        mock_curl.perform.assert_called_once()
        mock_curl.close.assert_called_once()


class TestOriginHasIpv6(unittest.TestCase):
    """Tests for _origin_has_ipv6 function."""

    def setUp(self):
        _AAAA_CACHE.clear()

    def tearDown(self):
        _AAAA_CACHE.clear()

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.socket.getaddrinfo")
    def test_returns_true_when_aaaa_records_exist(self, mock_getaddrinfo):
        """Test that _origin_has_ipv6 returns True when AAAA records are found."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0))
        ]
        self.assertTrue(_origin_has_ipv6("example.com"))
        mock_getaddrinfo.assert_called_once_with(
            "example.com", 443, socket.AF_INET6, socket.SOCK_STREAM
        )

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.socket.getaddrinfo")
    def test_returns_false_when_no_aaaa_records(self, mock_getaddrinfo):
        """Test that _origin_has_ipv6 returns False when no AAAA records exist."""
        mock_getaddrinfo.return_value = []
        self.assertFalse(_origin_has_ipv6("ipv4only.example.com"))

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.socket.getaddrinfo")
    def test_returns_false_on_gaierror(self, mock_getaddrinfo):
        """Test that _origin_has_ipv6 returns False on DNS resolution failure."""
        mock_getaddrinfo.side_effect = socket.gaierror("Name or service not known")
        self.assertFalse(_origin_has_ipv6("nonexistent.example.com"))

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.socket.getaddrinfo")
    def test_caches_result(self, mock_getaddrinfo):
        """Test that repeated calls for the same hostname use the cache."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0))
        ]
        self.assertTrue(_origin_has_ipv6("cached.example.com"))
        self.assertTrue(_origin_has_ipv6("cached.example.com"))
        mock_getaddrinfo.assert_called_once()

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.time.monotonic")
    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.socket.getaddrinfo")
    def test_cache_expires_after_ttl(self, mock_getaddrinfo, mock_monotonic):
        """Test that cached results expire after the TTL."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0))
        ]
        # First call at t=0
        mock_monotonic.return_value = 0.0
        self.assertTrue(_origin_has_ipv6("ttl.example.com"))
        # Second call at t=3601 (past 1-hour TTL)
        mock_monotonic.return_value = 3601.0
        self.assertTrue(_origin_has_ipv6("ttl.example.com"))
        self.assertEqual(2, mock_getaddrinfo.call_count)


class TestMaybeProxyUrl(unittest.TestCase):
    """Tests for _maybe_proxy_url function."""

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_returns_original_url_when_proxy_not_configured(self, mock_settings):
        """Test that _maybe_proxy_url returns the original URL when proxy is not set."""
        mock_settings.marqo_media_proxy_url = None
        result = _maybe_proxy_url("https://example.com/image.png")
        self.assertEqual("https://example.com/image.png", result)

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_constructs_proxy_url_correctly(self, mock_settings):
        """Test that _maybe_proxy_url constructs the correct proxy URL."""
        mock_settings.marqo_media_proxy_url = "https://media-proxy.marqo.workers.dev/proxy"
        result = _maybe_proxy_url("https://cdn.shopify.com/image.png")
        self.assertEqual(
            "https://media-proxy.marqo.workers.dev/proxy?url=https%3A%2F%2Fcdn.shopify.com%2Fimage.png",
            result,
        )

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_url_encodes_special_characters(self, mock_settings):
        """Test that special characters in the original URL are properly encoded."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        result = _maybe_proxy_url("https://example.com/path?q=hello world&size=large")
        self.assertIn("url=https%3A%2F%2Fexample.com", result)
        self.assertNotIn(" ", result.split("url=")[1])


class TestGetProxyHeaders(unittest.TestCase):
    """Tests for _get_proxy_headers function."""

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_returns_headers_when_both_configured(self, mock_settings):
        """Test that CF Access headers are returned when both ID and secret are set."""
        mock_settings.cf_access_client_id = "test-client-id"
        mock_settings.cf_access_client_secret = "test-client-secret"
        result = _get_proxy_headers()
        self.assertEqual(
            {
                "CF-Access-Client-Id": "test-client-id",
                "CF-Access-Client-Secret": "test-client-secret",
            },
            result,
        )

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_returns_empty_dict_when_id_missing(self, mock_settings):
        """Test that empty dict is returned when client ID is missing."""
        mock_settings.cf_access_client_id = None
        mock_settings.cf_access_client_secret = "test-client-secret"
        self.assertEqual({}, _get_proxy_headers())

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_returns_empty_dict_when_secret_missing(self, mock_settings):
        """Test that empty dict is returned when client secret is missing."""
        mock_settings.cf_access_client_id = "test-client-id"
        mock_settings.cf_access_client_secret = None
        self.assertEqual({}, _get_proxy_headers())

    @patch("inference_orchestrator.services.media_download_and_preprocess.image_download.settings")
    def test_returns_empty_dict_when_both_missing(self, mock_settings):
        """Test that empty dict is returned when both credentials are missing."""
        mock_settings.cf_access_client_id = None
        mock_settings.cf_access_client_secret = None
        self.assertEqual({}, _get_proxy_headers())


class TestIsProxyFailure(unittest.TestCase):
    """Tests for _is_proxy_failure function."""

    def _make_exc_with_pycurl_cause(self, curl_code, curl_msg="error"):
        """Helper to create an ImageDownloadError with a pycurl.error cause."""
        try:
            curl_exc = pycurl.error(curl_code, curl_msg)
            raise ImageDownloadError("download failed") from curl_exc
        except ImageDownloadError as exc:
            return exc

    def test_returns_true_for_pycurl_connect_error(self):
        exc = self._make_exc_with_pycurl_cause(pycurl.E_COULDNT_CONNECT)
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_true_for_pycurl_resolve_error(self):
        exc = self._make_exc_with_pycurl_cause(pycurl.E_COULDNT_RESOLVE_HOST)
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_true_for_pycurl_timeout_error(self):
        exc = self._make_exc_with_pycurl_cause(pycurl.E_OPERATION_TIMEDOUT)
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_true_for_http_502(self):
        exc = ImageDownloadError("media url `https://example.com` returned 502")
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_true_for_http_503(self):
        exc = ImageDownloadError("media url `https://example.com` returned 503")
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_true_for_http_504(self):
        exc = ImageDownloadError("media url `https://example.com` returned 504")
        self.assertTrue(_is_proxy_failure(exc))

    def test_returns_false_for_http_404(self):
        exc = ImageDownloadError("media url `https://example.com` returned 404")
        self.assertFalse(_is_proxy_failure(exc))

    def test_returns_false_for_http_403(self):
        exc = ImageDownloadError("media url `https://example.com` returned 403")
        self.assertFalse(_is_proxy_failure(exc))

    def test_returns_false_for_unrelated_pycurl_error(self):
        exc = self._make_exc_with_pycurl_cause(pycurl.E_ABORTED_BY_CALLBACK)
        self.assertFalse(_is_proxy_failure(exc))

    def test_returns_false_for_generic_error(self):
        exc = ImageDownloadError("some generic error")
        self.assertFalse(_is_proxy_failure(exc))


class TestDoDownload(unittest.TestCase):
    """Tests for _do_download function."""

    @patch("pycurl.Curl")
    def test_basic_download_success(self, mock_curl_class):
        """Test successful download through _do_download."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        result = _do_download("https://example.com/image.png", {}, 5000)
        self.assertIsInstance(result, BytesIO)
        mock_curl.perform.assert_called_once()
        mock_curl.close.assert_called_once()

    @patch("pycurl.Curl")
    def test_force_ipv6_sets_ipresolve(self, mock_curl_class):
        """Test that force_ipv6=True sets IPRESOLVE_V6 on the curl handle."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        _do_download("https://example.com/image.png", {}, 3000, force_ipv6=True)

        mock_curl.setopt.assert_any_call(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V6)

    @patch("pycurl.Curl")
    def test_force_ipv6_false_does_not_set_ipresolve(self, mock_curl_class):
        """Test that force_ipv6=False does not set IPRESOLVE."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        _do_download("https://example.com/image.png", {}, 3000, force_ipv6=False)

        ipresolve_calls = [
            call for call in mock_curl.setopt.call_args_list
            if call[0][0] == pycurl.IPRESOLVE
        ]
        self.assertEqual(0, len(ipresolve_calls))

    @patch("pycurl.Curl")
    def test_proxy_headers_merged_into_request(self, mock_curl_class):
        """Test that proxy headers are included in the request headers."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 200

        proxy_headers = {"CF-Access-Client-Id": "id123", "CF-Access-Client-Secret": "secret456"}
        _do_download("https://example.com/image.png", {}, 3000, proxy_headers=proxy_headers)

        header_calls = [
            call for call in mock_curl.setopt.call_args_list
            if call[0][0] == pycurl.HTTPHEADER
        ]
        self.assertEqual(1, len(header_calls))
        header_list = header_calls[0][0][1]
        self.assertIn("CF-Access-Client-Id: id123", header_list)
        self.assertIn("CF-Access-Client-Secret: secret456", header_list)

    @patch("pycurl.Curl")
    def test_fetch_url_used_for_curl_but_image_path_in_errors(self, mock_curl_class):
        """Test that fetch_url is used for the request but image_path appears in errors."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.getinfo.return_value = 404

        with self.assertRaises(ImageDownloadError) as ctx:
            _do_download(
                "https://origin.com/image.png", {}, 3000,
                fetch_url="https://proxy.example.com?url=https%3A%2F%2Forigin.com%2Fimage.png",
            )
        # Error message should reference the original URL, not the proxy URL
        self.assertIn("origin.com/image.png", str(ctx.exception))

    @patch("pycurl.Curl")
    def test_pycurl_error_preserves_cause(self, mock_curl_class):
        """Test that pycurl.error is preserved as __cause__ on ImageDownloadError."""
        mock_curl = MagicMock()
        mock_curl_class.return_value = mock_curl
        mock_curl.perform.side_effect = pycurl.error(
            pycurl.E_COULDNT_CONNECT, "Connection refused"
        )

        with self.assertRaises(ImageDownloadError) as ctx:
            _do_download("https://example.com/image.png", {}, 3000)
        self.assertIsInstance(ctx.exception.__cause__, pycurl.error)
        self.assertEqual(pycurl.E_COULDNT_CONNECT, ctx.exception.__cause__.args[0])


class TestDownloadImageFromUrlThreeTier(unittest.TestCase):
    """Tests for download_image_from_url three-tier strategy."""

    MODULE_PATH = "inference_orchestrator.services.media_download_and_preprocess.image_download"

    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_no_proxy_configured_uses_direct_download(self, mock_settings, mock_do_download):
        """Test that when proxy is not configured, _do_download is called directly."""
        mock_settings.marqo_media_proxy_url = None
        mock_do_download.return_value = BytesIO(b"image_data")

        result = download_image_from_url("https://example.com/image.png", {}, 3000)

        self.assertIsInstance(result, BytesIO)
        mock_do_download.assert_called_once_with(
            "https://example.com/image.png", {}, 3000, None
        )

    @patch(f"{MODULE_PATH}._origin_has_ipv6")
    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_tier1_direct_ipv6_success(self, mock_settings, mock_do_download, mock_has_ipv6):
        """Test Tier 1: direct IPv6 download succeeds when origin has AAAA records."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        mock_has_ipv6.return_value = True
        mock_do_download.return_value = BytesIO(b"image_data")

        result = download_image_from_url("https://cdn.shopify.com/image.png", {}, 3000)

        self.assertIsInstance(result, BytesIO)
        mock_do_download.assert_called_once_with(
            "https://cdn.shopify.com/image.png", {}, 3000, None,
            force_ipv6=True,
        )

    @patch(f"{MODULE_PATH}._get_proxy_headers")
    @patch(f"{MODULE_PATH}._maybe_proxy_url")
    @patch(f"{MODULE_PATH}._origin_has_ipv6")
    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_tier1_fails_falls_to_tier2(
        self, mock_settings, mock_do_download, mock_has_ipv6, mock_proxy_url, mock_proxy_headers
    ):
        """Test that Tier 1 failure falls through to Tier 2."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        mock_has_ipv6.return_value = True
        mock_proxy_url.return_value = "https://proxy.example.com?url=encoded"
        mock_proxy_headers.return_value = {"CF-Access-Client-Id": "id"}

        # First call (Tier 1) fails, second call (Tier 2) succeeds
        mock_do_download.side_effect = [
            ImageDownloadError("IPv6 connection failed"),
            BytesIO(b"image_data"),
        ]

        result = download_image_from_url("https://example.com/image.png", {}, 3000)

        self.assertIsInstance(result, BytesIO)
        self.assertEqual(2, mock_do_download.call_count)
        # Second call should be to the proxy
        second_call = mock_do_download.call_args_list[1]
        self.assertEqual("https://proxy.example.com?url=encoded", second_call.kwargs["fetch_url"])

    @patch(f"{MODULE_PATH}._get_proxy_headers")
    @patch(f"{MODULE_PATH}._maybe_proxy_url")
    @patch(f"{MODULE_PATH}._origin_has_ipv6")
    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_tier2_proxy_success_for_ipv4_only_origin(
        self, mock_settings, mock_do_download, mock_has_ipv6, mock_proxy_url, mock_proxy_headers
    ):
        """Test Tier 2: proxy download succeeds when origin is IPv4-only."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        mock_has_ipv6.return_value = False
        mock_proxy_url.return_value = "https://proxy.example.com?url=encoded"
        mock_proxy_headers.return_value = {"CF-Access-Client-Id": "id"}
        mock_do_download.return_value = BytesIO(b"image_data")

        result = download_image_from_url("https://ipv4only.example.com/image.png", {}, 3000)

        self.assertIsInstance(result, BytesIO)
        mock_do_download.assert_called_once_with(
            "https://ipv4only.example.com/image.png", {}, 3000, None,
            proxy_headers={"CF-Access-Client-Id": "id"},
            fetch_url="https://proxy.example.com?url=encoded",
        )

    @patch(f"{MODULE_PATH}._is_proxy_failure")
    @patch(f"{MODULE_PATH}._get_proxy_headers")
    @patch(f"{MODULE_PATH}._maybe_proxy_url")
    @patch(f"{MODULE_PATH}._origin_has_ipv6")
    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_tier2_proxy_failure_falls_to_tier3(
        self, mock_settings, mock_do_download, mock_has_ipv6,
        mock_proxy_url, mock_proxy_headers, mock_is_proxy_failure
    ):
        """Test that Tier 2 proxy failure falls back to Tier 3 (direct IPv4)."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        mock_has_ipv6.return_value = False
        mock_proxy_url.return_value = "https://proxy.example.com?url=encoded"
        mock_proxy_headers.return_value = {}
        mock_is_proxy_failure.return_value = True

        proxy_exc = ImageDownloadError("media url `https://example.com` returned 502")
        mock_do_download.side_effect = [proxy_exc, BytesIO(b"image_data")]

        result = download_image_from_url("https://example.com/image.png", {}, 3000)

        self.assertIsInstance(result, BytesIO)
        self.assertEqual(2, mock_do_download.call_count)
        # Third-tier call should be direct (no proxy_headers or fetch_url)
        third_call = mock_do_download.call_args_list[1]
        self.assertEqual(
            ("https://example.com/image.png", {}, 3000, None),
            third_call[0],
        )

    @patch(f"{MODULE_PATH}._is_proxy_failure")
    @patch(f"{MODULE_PATH}._get_proxy_headers")
    @patch(f"{MODULE_PATH}._maybe_proxy_url")
    @patch(f"{MODULE_PATH}._origin_has_ipv6")
    @patch(f"{MODULE_PATH}._do_download")
    @patch(f"{MODULE_PATH}.settings")
    def test_tier2_origin_error_does_not_fallback(
        self, mock_settings, mock_do_download, mock_has_ipv6,
        mock_proxy_url, mock_proxy_headers, mock_is_proxy_failure
    ):
        """Test that origin errors from Tier 2 are raised, not retried."""
        mock_settings.marqo_media_proxy_url = "https://proxy.example.com"
        mock_has_ipv6.return_value = False
        mock_proxy_url.return_value = "https://proxy.example.com?url=encoded"
        mock_proxy_headers.return_value = {}
        mock_is_proxy_failure.return_value = False

        mock_do_download.side_effect = ImageDownloadError(
            "media url `https://example.com` returned 404"
        )

        with self.assertRaises(ImageDownloadError) as ctx:
            download_image_from_url("https://example.com/image.png", {}, 3000)
        self.assertIn("404", str(ctx.exception))
        # Should only be called once (Tier 2), no fallback to Tier 3
        mock_do_download.assert_called_once()

    def test_invalid_timeout_raises_before_any_download(self):
        """Test that invalid timeout raises InternalServerError immediately."""
        with self.assertRaises(InternalServerError):
            download_image_from_url("https://example.com/image.png", {}, "not_int")


class TestEncodeUrl(unittest.TestCase):
    """Tests for encode_url function."""

    def test_encode_url_basic(self):
        """Test that encode_url properly encodes URLs."""
        url = "https://example.com/image with spaces.png"
        result = encode_url(url)
        self.assertIsInstance(result, str)
        self.assertNotIn(" ", result)

    def test_encode_url_already_encoded(self):
        """Test that encode_url handles already encoded URLs."""
        url = "https://example.com/image.png"
        result = encode_url(url)
        self.assertEqual(url, result)


class TestDownloadMediaFromUrl(unittest.TestCase):
    """Tests for download_media_from_url function."""

    @patch(
        "inference_orchestrator.services.media_download_and_preprocess.image_download.download_image_from_url"
    )
    def test_download_media_from_url_calls_download_image(self, mock_download):
        """Test that download_media_from_url delegates to download_image_from_url."""
        mock_download.return_value = BytesIO()

        result = download_media_from_url(
            "https://example.com/media.mp4", {"header": "value"}, 5000, "video"
        )

        mock_download.assert_called_once_with(
            "https://example.com/media.mp4", {"header": "value"}, 5000, "video"
        )
        self.assertIsInstance(result, BytesIO)


if __name__ == "__main__":
    unittest.main()
