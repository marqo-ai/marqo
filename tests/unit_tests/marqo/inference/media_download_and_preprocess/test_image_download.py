import base64
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image, UnidentifiedImageError

from marqo.inference.media_download_and_preprocess.image_download import (
    format_and_load_CLIP_image,
    load_image_from_path,
)


class TestBase64ImageSupport(unittest.TestCase):
    def setUp(self):
        """Create test images for use in tests."""
        # Create a small test image (2x2 pixels)
        self.test_image = Image.new("RGB", (2, 2), color="red")
        buffer = BytesIO()
        self.test_image.save(buffer, format="PNG")
        self.test_image_bytes = buffer.getvalue()
        self.test_base64_data = base64.b64encode(self.test_image_bytes).decode("utf-8")
        self.test_data_url = f"data:image/png;base64,{self.test_base64_data}"

    def test_load_image_from_path_with_base64_data_url(self):
        """Test loading base64 image from data URL format through public API."""
        img = load_image_from_path(self.test_data_url, {})
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (2, 2))

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

    def test_load_image_from_path_handles_various_base64_formats(self):
        """Test that load_image_from_path handles different base64 formats."""
        # Test with data URL format
        result = load_image_from_path(self.test_data_url, {})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (2, 2))

        # Test with different MIME types
        jpeg_image = Image.new("RGB", (3, 3), color="blue")
        buffer = BytesIO()
        jpeg_image.save(buffer, format="JPEG")
        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        jpeg_data_url = f"data:image/jpeg;base64,{jpeg_base64}"

        result_jpeg = load_image_from_path(jpeg_data_url, {})
        self.assertIsInstance(result_jpeg, Image.Image)
        self.assertEqual(result_jpeg.size, (3, 3))

    def test_format_and_load_CLIP_image_base64(self):
        """Test that format_and_load_CLIP_image handles base64 images end-to-end."""
        # Test with actual base64 data URL
        result = format_and_load_CLIP_image(self.test_data_url, {})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (2, 2))

    def test_load_image_from_path_base64_precedence(self):
        """Test that base64 detection takes precedence over file/URL checks."""
        # When a string is a valid base64 data URL, it should be processed as base64
        # even if the system might try to interpret it as a file path
        result = load_image_from_path(self.test_data_url, {})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (2, 2))

        # Test that actual file paths work correctly (not base64)
        with patch("os.path.isfile") as mock_isfile:
            with patch("PIL.Image.open") as mock_open:
                mock_isfile.return_value = True
                mock_image = MagicMock()
                mock_open.return_value = mock_image

                # This should use file loading, not base64
                result = load_image_from_path("/real/file/path.png", {})

                mock_isfile.assert_called_with("/real/file/path.png")
                mock_open.assert_called_once()
                self.assertEqual(result, mock_image)


if __name__ == "__main__":
    unittest.main()
