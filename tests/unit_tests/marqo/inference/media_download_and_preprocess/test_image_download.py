import unittest
from unittest.mock import patch, MagicMock
import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from marqo.core.inference.modality_utils import is_base64_image
from marqo.inference.media_download_and_preprocess.image_download import (
    _load_base64_image, load_image_from_path, format_and_load_CLIP_image
)


class TestBase64ImageSupport(unittest.TestCase):
    
    def setUp(self):
        """Create test images for use in tests."""
        # Create a small test image (2x2 pixels)
        self.test_image = Image.new('RGB', (2, 2), color='red')
        buffer = BytesIO()
        self.test_image.save(buffer, format='PNG')
        self.test_image_bytes = buffer.getvalue()
        self.test_base64_data = base64.b64encode(self.test_image_bytes).decode('utf-8')
        self.test_data_url = f"data:image/png;base64,{self.test_base64_data}"

    def test_is_base64_image_data_url_format(self):
        """Test recognition of data URL format base64 images."""
        self.assertTrue(is_base64_image(self.test_data_url))
        
        # Test different formats
        jpeg_data_url = f"data:image/jpeg;base64,{self.test_base64_data}"
        self.assertTrue(is_base64_image(jpeg_data_url))

    def test_is_base64_image_plain_base64(self):
        """Test recognition of plain base64 images."""
        with patch('magic.from_buffer') as mock_magic:
            mock_magic.return_value = 'image/png'
            self.assertTrue(is_base64_image(self.test_base64_data))

    def test_is_base64_image_invalid_cases(self):
        """Test rejection of invalid cases."""
        # Short string
        self.assertFalse(is_base64_image("short"))
        
        # Non-base64 string
        self.assertFalse(is_base64_image("not_base64_at_all" * 10))
        
        # Non-image content
        with patch('magic.from_buffer') as mock_magic:
            mock_magic.return_value = 'text/plain'
            self.assertFalse(is_base64_image("VGVzdCB0ZXh0" * 10))

    def test_load_base64_image_data_url(self):
        """Test loading base64 image from data URL format."""
        img = _load_base64_image(self.test_data_url)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (2, 2))

    def test_load_base64_image_plain_base64(self):
        """Test loading base64 image from plain base64 string."""
        img = _load_base64_image(self.test_base64_data)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (2, 2))

    def test_load_base64_image_invalid_data(self):
        """Test error handling for invalid base64 data."""
        with self.assertRaises(UnidentifiedImageError):
            _load_base64_image("invalid_base64!!!")
        
        with self.assertRaises(UnidentifiedImageError):
            _load_base64_image("data:image/png;base64,invalid!!!")

    def test_load_image_from_path_base64_data_url(self):
        """Test that load_image_from_path handles base64 data URLs."""
        with patch('marqo.inference.media_download_and_preprocess.image_download.is_base64_image') as mock_is_base64:
            with patch('marqo.inference.media_download_and_preprocess.image_download._load_base64_image') as mock_load:
                mock_is_base64.return_value = True
                mock_load.return_value = self.test_image
                
                result = load_image_from_path(self.test_data_url, {})
                
                mock_is_base64.assert_called_once_with(self.test_data_url)
                mock_load.assert_called_once_with(self.test_data_url)
                self.assertEqual(result, self.test_image)

    def test_load_image_from_path_base64_plain(self):
        """Test that load_image_from_path handles plain base64 strings."""
        with patch('marqo.inference.media_download_and_preprocess.image_download.is_base64_image') as mock_is_base64:
            with patch('marqo.inference.media_download_and_preprocess.image_download._load_base64_image') as mock_load:
                mock_is_base64.return_value = True
                mock_load.return_value = self.test_image
                
                result = load_image_from_path(self.test_base64_data, {})
                
                mock_is_base64.assert_called_once_with(self.test_base64_data)
                mock_load.assert_called_once_with(self.test_base64_data)
                self.assertEqual(result, self.test_image)

    def test_format_and_load_CLIP_image_base64(self):
        """Test that format_and_load_CLIP_image handles base64 images."""
        with patch('marqo.inference.media_download_and_preprocess.image_download.load_image_from_path') as mock_load:
            mock_load.return_value = self.test_image
            
            result = format_and_load_CLIP_image(self.test_data_url, {})
            
            mock_load.assert_called_once_with(self.test_data_url, {})
            self.assertEqual(result, self.test_image)

    def test_load_image_from_path_precedence(self):
        """Test that base64 detection takes precedence over file/URL checks."""
        # Create a string that might look like a file path but is actually base64
        fake_path = "some/fake/path.png"
        
        with patch('marqo.inference.media_download_and_preprocess.image_download.is_base64_image') as mock_is_base64:
            with patch('marqo.inference.media_download_and_preprocess.image_download._load_base64_image') as mock_load:
                with patch('os.path.isfile') as mock_isfile:
                    mock_is_base64.return_value = True
                    mock_load.return_value = self.test_image
                    mock_isfile.return_value = True  # Pretend it's a real file
                    
                    result = load_image_from_path(fake_path, {})
                    
                    # Base64 check should happen first and prevent file check
                    mock_is_base64.assert_called_once_with(fake_path)
                    mock_load.assert_called_once_with(fake_path)
                    # os.path.isfile should not be called due to base64 precedence
                    mock_isfile.assert_not_called()
                    self.assertEqual(result, self.test_image)


if __name__ == '__main__':
    unittest.main() 