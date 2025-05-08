import unittest
from unittest.mock import patch, MagicMock

import requests

from marqo.core.inference.api import Modality
from marqo.core.inference.modality_utils import fetch_content_sample, infer_modality, \
    _infer_modality_based_on_extension, \
    get_url_file_extension


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

    @patch('marqo.core.inference.modality_utils.validate_url')
    @patch('marqo.core.inference.modality_utils.fetch_content_sample')
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

    def test_infer_modality_extension_with_query_parameters(self):
        test_cases = [
            # Correct cases with query parameters
            ("https://example.com/image.jpg?query=string", Modality.IMAGE, "Simple image URL with one query param"),
            ("https://example.com/video.mp4?foo=bar&baz=qux", Modality.VIDEO, "Video URL with multiple query params"),
            ("https://example.com/audio.mp3?abc=def&123=456", Modality.AUDIO,
             "Audio URL with numeric and alpha query params"),

            # Correct cases with more complex URLs
            ("https://example.com/photo.jpeg?weirdparam=??&another=##", Modality.IMAGE,
             "Valid image with strange query parameters"),
            ("https://example.com/sound.mp3?", Modality.AUDIO, "Valid audio with empty query string"),
            ("https://example.com/clip.mp4#fragment", Modality.VIDEO, "Video URL with fragment identifier"),

            # Edge cases: missing or no extension
            ("https://example.com/file.unknown?param=test", None, "Unknown extension should return None"),
            ("https://example.com/no_extension?query=data", None, "URL with no extension should return None"),
            ("https://example.com/imagejpg?query=string", None,
             "URL with incorrect extension format (missing dot) should return None"),
        ]

        for url, expected_modality, message in test_cases:
            with self.subTest(msg=message, url=url):
                inferred_modality = _infer_modality_based_on_extension(get_url_file_extension(url))
                self.assertEqual(expected_modality, inferred_modality)

    def test_infer_modality_no_extension_found(self):
        """A test to ensure if the extension is not found, we go to the mime type"""
        url = "https://example.com/file.unknown"

        with patch('marqo.core.inference.modality_utils.fetch_content_sample') as mock_fetch, \
                patch('marqo.core.inference.modality_utils.magic.from_buffer', return_value="audio/mpeg"), \
                patch('marqo.core.inference.modality_utils._infer_modality_based_on_mime_type') as mock_infer_on_mime:
            mock_fetch.return_value = MagicMock()
            _ = infer_modality(url)
            mock_infer_on_mime.assert_called_once_with("audio/mpeg")

    def test_infer_modality_proper_extension_found(self):
        """A test to ensure if the extension is found, we do not go to the mime type."""
        url = "https://example.com/file.mp3"

        with patch('marqo.core.inference.modality_utils._infer_modality_based_on_extension') as mock_infer_on_extension, \
                patch('marqo.core.inference.modality_utils.fetch_content_sample') as mock_fetch, \
                patch('marqo.core.inference.modality_utils.magic.from_buffer') as mock_magic, \
                patch('marqo.core.inference.modality_utils._infer_modality_based_on_mime_type') as mock_infer_on_mime:

            _ = infer_modality(url)
            mock_infer_on_extension.assert_called_once_with("mp3")
            mock_fetch.assert_not_called()
            mock_magic.assert_not_called()
            mock_infer_on_mime.assert_not_called()

    def test_infer_modality_receive_bytes_code_path(self):
        """A test to ensure if bytes is received, we skip extension and mime download, but mime check on the bytes."""
        bytes = b"test"

        with patch('marqo.core.inference.modality_utils._infer_modality_based_on_extension') as mock_infer_on_extension, \
                patch('marqo.core.inference.modality_utils.fetch_content_sample') as mock_fetch, \
                patch('marqo.core.inference.modality_utils.magic.from_buffer', return_value="image/jpeg") as mock_magic, \
                patch('marqo.core.inference.modality_utils._infer_modality_based_on_mime_type') as mock_infer_on_mime:

            _ = infer_modality(bytes)
            mock_infer_on_extension.assert_not_called()
            mock_fetch.assert_not_called()
            mock_magic.assert_called_once_with(bytes, mime=True)
            mock_infer_on_mime.assert_called_once_with("image/jpeg")
