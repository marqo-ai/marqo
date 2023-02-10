import PIL

from marqo.s2_inference import clip_utils, types
import unittest


class TestEncoding(unittest.TestCase):

    def test_load_image_from_path_timeout(self):
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        # should be fine on regular timeout:
        img = clip_utils.load_image_from_path(good_url)
        assert isinstance(img, types.ImageType)
        try:
            # should definitely timeout:
            img = clip_utils.load_image_from_path(good_url, timeout=0.0000001)
            raise AssertionError
        except PIL.UnidentifiedImageError:
            pass
