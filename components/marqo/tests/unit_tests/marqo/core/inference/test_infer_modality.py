import unittest
import os
import socket
import threading
from unittest.mock import patch, MagicMock
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import requests
from PIL import Image

from marqo.tensor_search.enums import EnvVars
from marqo.core.inference.api import MediaDownloadError, Modality
from marqo.core.inference.modality_utils import fetch_content_sample, infer_modality, \
    _infer_modality_based_on_extension, \
    get_url_file_extension, is_base64_image


def _resolves_to(*addresses):
    """Build a getaddrinfo replacement so destination checks do not depend on real DNS."""

    def _getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, port)) for address in addresses]

    return _getaddrinfo


def _non_redirect_response(**attributes):
    """Build a mock response that the redirect loop treats as the final hop."""
    response = MagicMock()
    response.is_redirect = False
    for name, value in attributes.items():
        setattr(response, name, value)
    return response


class TestMultimodalUtils(unittest.TestCase):

    def setUp(self):
        # A publicly routable address, so these tests exercise the allowed path of the
        # destination check without reaching the network.
        patcher = patch('socket.getaddrinfo', side_effect=_resolves_to('93.184.216.34'))
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch('requests.get')
    def test_fetch_content_sample(self, mock_get):
        url = "https://example.com/sample.txt"
        mock_get.return_value = _non_redirect_response(iter_content=MagicMock(return_value=[b'sample content']))

        with fetch_content_sample(url) as sample:
            self.assertEqual(sample.read(), b'sample content')

    @patch('requests.get')
    def test_fetch_content_sample_large_size(self, mock_get):
        url = "https://example.com/large_sample.txt"
        mock_get.return_value = _non_redirect_response(
            iter_content=MagicMock(return_value=[b'a' * 5000, b'b' * 5000, b'c' * 5000]))

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

    def test_is_base64_image_data_url_format(self):
        """Test recognition of data URL format base64 images."""
        # Create a small test image (1x1 red pixel PNG)
        img = Image.new('RGB', (1, 1), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Test data URL format
        data_url = f"data:image/png;base64,{base64_data}"
        self.assertTrue(is_base64_image(data_url))
        
        # Test different image formats
        data_url_jpeg = f"data:image/jpeg;base64,{base64_data}"
        self.assertTrue(is_base64_image(data_url_jpeg))

    def test_is_base64_image_invalid_cases(self):
        """Test rejection of invalid base64 image cases."""
        self.assertFalse(is_base64_image("short"))
        self.assertFalse(is_base64_image("not_base64_at_all" * 10))
        self.assertFalse(is_base64_image("data:text/plain;base64,VGVzdA=="))
        
    def test_infer_modality_base64_images(self):
        """Test that infer_modality correctly identifies base64 images."""
        # Create a small test image
        img = Image.new('RGB', (1, 1), color='purple')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Test data URL format
        data_url = f"data:image/png;base64,{base64_data}"
        self.assertEqual(infer_modality(data_url), Modality.IMAGE)


class _StaticContentHandler(BaseHTTPRequestHandler):
    """Serves a small PNG so that a completed fetch is distinguishable from a refused one."""

    protocol_version = "HTTP/1.0"
    body = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        """Silence the default stderr request log."""


def _make_redirect_handler(location):
    """Build a handler that answers every request with a 302 to `location`."""

    class _RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, *args):
            """Silence the default stderr request log."""

    return _RedirectHandler


def _serve(handler) -> ThreadingHTTPServer:
    """Start `handler` on an ephemeral loopback port and return the running server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestFetchContentSampleDestinationChecks(unittest.TestCase):
    """The MIME sniffing fetch must refuse destinations that are not publicly routable.

    `infer_modality` reaches this fetch for any URL without a recognised file extension,
    on both the add_documents and the search path, so it is reachable by an unauthenticated
    caller and needs the same destination check as the media download itself.
    """

    def setUp(self):
        self.server = _serve(_StaticContentHandler)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def test_fetch_from_loopback_address_is_refused(self):
        """A loopback URL must not be fetched, even though the server would answer it."""
        with self.assertRaises(MediaDownloadError) as context:
            with fetch_content_sample(f"http://127.0.0.1:{self.port}/sample"):
                pass

        self.assertIn("not publicly routable", str(context.exception))

    def test_infer_modality_refuses_loopback_url_without_extension(self):
        """The refusal must surface through infer_modality rather than being reported as TEXT."""
        with self.assertRaises(MediaDownloadError) as context:
            infer_modality(f"http://127.0.0.1:{self.port}/sample")

        self.assertIn("not publicly routable", str(context.exception))

    def test_fetch_refuses_redirect_to_internal_address(self):
        """A reachable first hop must not be able to redirect the fetch to an internal address."""
        redirect_server = _serve(_make_redirect_handler("http://169.254.169.254/latest/meta-data/"))
        self.addCleanup(redirect_server.server_close)
        self.addCleanup(redirect_server.shutdown)
        url = f"http://127.0.0.1:{redirect_server.server_address[1]}/start"

        with patch.dict(os.environ, {EnvVars.MARQO_MEDIA_DOWNLOAD_ALLOWED_NETWORKS: "127.0.0.0/8"}):
            with self.assertRaises(MediaDownloadError) as context:
                with fetch_content_sample(url):
                    pass

        self.assertIn("not publicly routable", str(context.exception))
        self.assertIn("169.254.169.254", str(context.exception))

    def test_fetch_allows_destination_named_in_allowed_networks(self):
        """An operator that serves media from a private network can name it and be served."""
        url = f"http://127.0.0.1:{self.port}/sample"

        with patch.dict(os.environ, {EnvVars.MARQO_MEDIA_DOWNLOAD_ALLOWED_NETWORKS: "127.0.0.0/8"}):
            with fetch_content_sample(url) as sample:
                self.assertEqual(_StaticContentHandler.body, sample.read())

    def test_fetch_refuses_non_http_scheme(self):
        """Only http and https are fetched, so a file URL never reaches the HTTP client."""
        with self.assertRaises(MediaDownloadError) as context:
            with fetch_content_sample("file:///etc/hostname"):
                pass

        self.assertIn("only downloads media over http and https", str(context.exception))

    def test_fetch_refuses_a_redirect_loop(self):
        """A redirect chain that never terminates is bounded rather than followed forever."""
        server = _serve(_make_redirect_handler("/again"))
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/start"

        with patch.dict(os.environ, {EnvVars.MARQO_MEDIA_DOWNLOAD_ALLOWED_NETWORKS: "127.0.0.0/8"}):
            with self.assertRaises(MediaDownloadError) as context:
                with fetch_content_sample(url):
                    pass

        self.assertIn("exceeded", str(context.exception))
