import base64
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
    _load_base64_image,
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
