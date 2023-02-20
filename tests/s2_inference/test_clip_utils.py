import PIL
from marqo.s2_inference import clip_utils, types
import unittest
from unittest import mock
import requests


class TestEncoding(unittest.TestCase):

    def test_load_image_from_path_timeout(self):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # should be fine on regular timeout:
        img = clip_utils.load_image_from_path(good_url, {})
        assert isinstance(img, types.ImageType)
        try:
            # should definitely timeout:
            clip_utils.load_image_from_path(good_url, {}, timeout=0.0000001)
            raise AssertionError
        except PIL.UnidentifiedImageError:
            pass

    def test_load_image_from_path_all_req_errors(self):
        """Do we catch other download errors?
        The errors tested inherit from requests.exceptions.RequestException
        """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # should be fine on regular timeout:
        clip_utils.load_image_from_path(good_url, {})

        for err in [requests.exceptions.ReadTimeout, requests.exceptions.HTTPError]:
            mock_get = mock.MagicMock()
            mock_get.side_effect = err

            @mock.patch('requests.get', mock_get)
            def run():
                try:
                    clip_utils.load_image_from_path(good_url, {})
                    raise AssertionError
                except PIL.UnidentifiedImageError:
                    pass
                return True

            run()
