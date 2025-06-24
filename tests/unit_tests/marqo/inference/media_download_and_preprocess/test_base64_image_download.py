import base64
import binascii
import unittest
from unittest.mock import patch

import PIL.Image
import numpy as np

from marqo.inference.media_download_and_preprocess.image_download import (
    _is_base64_image,
    _decode_base64_image,
    _is_image,
    format_and_load_CLIP_image
)
from marqo.s2_inference.errors import ImageDownloadError
from PIL import UnidentifiedImageError


class TestBase64ImageDownload(unittest.TestCase):

    def test_is_base64_image_data_url_format(self):
        """Test detection of base64 data URL format."""
        valid_data_urls = [
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs="
        ]
        
        for data_url in valid_data_urls:
            with self.subTest(data_url=data_url):
                self.assertTrue(_is_base64_image(data_url))

    def test_is_base64_image_invalid_data_url(self):
        """Test rejection of invalid data URLs."""
        invalid_data_urls = [
            "data:image/jpeg;/9j/4AAQSkZJRgABAQEAYABgAAD/",  # Missing base64 marker
            "data:text/plain;base64,SGVsbG8gV29ybGQ=",  # Not an image
            "http://example.com/image.jpg",  # Not a data URL
            "data:image/png;base64",  # Missing data
            ""
        ]
        
        for data_url in invalid_data_urls:
            with self.subTest(data_url=data_url):
                self.assertFalse(_is_base64_image(data_url))

    def test_is_base64_image_plain_base64(self):
        """Test detection of plain base64 strings."""
        # Valid base64 string (1x1 PNG image)
        valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        self.assertTrue(_is_base64_image(valid_base64))
        
        # Invalid cases
        invalid_cases = [
            "short",  # Too short
            "notbase64!@#$%",  # Invalid characters
            "validlength123",  # Not a multiple of 4
            123,  # Not a string
            None
        ]
        
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                self.assertFalse(_is_base64_image(invalid))

    def test_decode_base64_image_data_url(self):
        """Test decoding of base64 data URL to PIL Image."""
        # 1x1 PNG image data URL
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        img = _decode_base64_image(data_url)
        self.assertIsInstance(img, PIL.Image.Image)
        self.assertEqual(img.size, (1, 1))
        self.assertEqual(img.format, 'PNG')

    def test_decode_base64_image_plain_base64(self):
        """Test decoding of plain base64 string to PIL Image."""
        # 1x1 PNG image base64 string
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        img = _decode_base64_image(base64_str)
        self.assertIsInstance(img, PIL.Image.Image)
        self.assertEqual(img.size, (1, 1))
        self.assertEqual(img.format, 'PNG')

    def test_decode_base64_image_invalid_data_url(self):
        """Test error handling for invalid data URL format."""
        invalid_data_url = "data:image/jpeg;invalid_marker,/9j/4AAQ"
        
        with self.assertRaises(UnidentifiedImageError) as cm:
            _decode_base64_image(invalid_data_url)
        
        self.assertIn("Base64 data URL must contain ';base64,' marker", str(cm.exception))

    def test_decode_base64_image_invalid_base64(self):
        """Test error handling for invalid base64 content."""
        invalid_cases = [
            "invalid!@#base64",
            "data:image/png;base64,invalid_base64_content",
            "data:image/jpeg;base64,not_valid_base64_data"
        ]
        
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(UnidentifiedImageError):
                    _decode_base64_image(invalid)

    def test_decode_base64_image_not_an_image(self):
        """Test error handling when base64 decodes but is not a valid image."""
        # Valid base64 but not an image (text content)
        text_base64 = "data:image/jpeg;base64," + base64.b64encode(b"Hello World").decode()
        
        with self.assertRaises(UnidentifiedImageError):
            _decode_base64_image(text_base64)

    def test_is_image_recognizes_base64(self):
        """Test that _is_image function recognizes base64 images."""
        base64_cases = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ]
        
        for base64_str in base64_cases:
            with self.subTest(base64=base64_str[:50] + "..."):
                self.assertTrue(_is_image(base64_str))

    def test_is_image_list_with_base64(self):
        """Test that _is_image function works with lists containing base64 images."""
        base64_list = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ]
        
        self.assertTrue(_is_image(base64_list))

    def test_format_and_load_CLIP_image_base64_data_url(self):
        """Test format_and_load_CLIP_image with base64 data URL."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        img = format_and_load_CLIP_image(data_url, {})
        self.assertIsInstance(img, PIL.Image.Image)
        self.assertEqual(img.size, (1, 1))

    def test_format_and_load_CLIP_image_plain_base64(self):
        """Test format_and_load_CLIP_image with plain base64 string."""
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        img = format_and_load_CLIP_image(base64_str, {})
        self.assertIsInstance(img, PIL.Image.Image)
        self.assertEqual(img.size, (1, 1))

    def test_format_and_load_CLIP_image_fallback_to_path(self):
        """Test that format_and_load_CLIP_image falls back to path loading for non-base64 strings."""
        with patch('marqo.inference.media_download_and_preprocess.image_download.load_image_from_path') as mock_load:
            mock_load.return_value = PIL.Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            
            # Non-base64 string should use path loading
            non_base64_str = "http://example.com/image.jpg"
            img = format_and_load_CLIP_image(non_base64_str, {})
            
            mock_load.assert_called_once_with(non_base64_str, {})
            self.assertIsInstance(img, PIL.Image.Image)

    def test_decode_large_base64_image(self):
        """Test decoding of a larger base64 image."""
        # Create a small test image
        test_img = PIL.Image.new('RGB', (10, 10), color='red')
        
        # Convert to base64
        from io import BytesIO
        buffer = BytesIO()
        test_img.save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode()
        
        # Test decoding
        decoded_img = _decode_base64_image(base64_str)
        self.assertIsInstance(decoded_img, PIL.Image.Image)
        self.assertEqual(decoded_img.size, (10, 10))

    def test_edge_cases(self):
        """Test various edge cases for base64 image handling."""
        # Empty string
        self.assertFalse(_is_base64_image(""))
        
        # Very short base64-looking strings
        self.assertFalse(_is_base64_image("abc="))
        
        # Malformed data URL
        self.assertFalse(_is_base64_image("data:image/png;base64"))
        
        # Non-string input
        self.assertFalse(_is_base64_image(None))
        self.assertFalse(_is_base64_image(123))
        self.assertFalse(_is_base64_image(['not', 'a', 'string']))


if __name__ == '__main__':
    unittest.main()