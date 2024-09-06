import unittest
from unittest.mock import patch, MagicMock
import requests
import io
from marqo.s2_inference.multimodal_model_load import Modality, infer_modality, fetch_content_sample

class TestMultimodalUtils(unittest.TestCase):

    @patch('requests.get')
    def test_fetch_content_sample(self, mock_get):
        url = "https://example.com/sample.txt"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'sample content']
        mock_get.return_value = mock_response

        with fetch_content_sample(url) as sample:
            self.assertEqual(sample.read(), b'sample content')

    @patch('requests.get')
    def test_fetch_content_sample_large_size(self, mock_get):
        url = "https://example.com/large_sample.txt"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'a' * 5000, b'b' * 5000, b'c' * 5000]
        mock_get.return_value = mock_response

        with fetch_content_sample(url, sample_size=15000) as sample:
            content = sample.read()
            self.assertEqual(len(content), 15000)
            self.assertTrue(content.startswith(b'a' * 5000 + b'b' * 5000))

    @patch('requests.get')
    def test_fetch_content_sample_network_error(self, mock_get):
        url = "https://example.com/error.txt"
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            with fetch_content_sample(url):
                pass

    def test_infer_modality_text(self):
        self.assertEqual(infer_modality("This is a sample text."), Modality.TEXT)
        self.assertEqual(infer_modality(""), Modality.TEXT)  # Empty string

    def test_infer_modality_url_with_extension(self):
        self.assertEqual(infer_modality("https://example.com/image.jpg"), Modality.IMAGE)
        self.assertEqual(infer_modality("https://example.com/video.mp4"), Modality.VIDEO)
        self.assertEqual(infer_modality("https://example.com/audio.mp3"), Modality.AUDIO)

    @patch('marqo.s2_inference.multimodal_model_load.validate_url')
    @patch('marqo.s2_inference.multimodal_model_load.fetch_content_sample')
    def test_infer_modality_url_without_extension(self, mock_fetch, mock_validate):
        mock_validate.return_value = True
        mock_sample = MagicMock()
        mock_fetch.return_value.__enter__.return_value = mock_sample

        with patch('magic.from_buffer') as mock_magic:
            mock_magic.return_value = 'image/jpeg'
            self.assertEqual(infer_modality("https://example.com/image"), Modality.IMAGE)

            mock_magic.return_value = 'video/mp4'
            self.assertEqual(infer_modality("https://example.com/video"), Modality.VIDEO)

            mock_magic.return_value = 'audio/mpeg'
            self.assertEqual(infer_modality("https://example.com/audio"), Modality.AUDIO)

    def test_infer_modality_invalid_url(self):
        self.assertEqual(infer_modality("not_a_url"), Modality.TEXT)

    def test_infer_modality_bytes(self):
        with patch('magic.from_buffer') as mock_magic:
            mock_magic.return_value = 'image/jpeg'
            self.assertEqual(infer_modality(b'\xff\xd8\xff'), Modality.IMAGE)

            mock_magic.return_value = 'video/mp4'
            self.assertEqual(infer_modality(b'\x00\x00\x00 ftyp'), Modality.VIDEO)

            mock_magic.return_value = 'audio/mpeg'
            self.assertEqual(infer_modality(b'ID3'), Modality.AUDIO)

            mock_magic.return_value = 'text/plain'
            self.assertEqual(infer_modality(b'plain text'), Modality.TEXT)

    def test_infer_modality_list_of_strings(self):
        self.assertEqual(infer_modality(["text1", "text2"]), Modality.TEXT)

    def test_infer_modality_empty_bytes(self):
        self.assertEqual(infer_modality(b''), Modality.TEXT)

if __name__ == '__main__':
    unittest.main()