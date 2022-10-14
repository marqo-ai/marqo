from marqo.tensor_search.utils import content_routering
from tests.marqo_test import MarqoTestCase
import unittest
import numpy as np


class TestContentRoutering(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def validation_test(self):
        validation_test_example = {

            # we should pass all the examples in this dictionary

            # A straight image https in jpg format
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",

            # A straight video https in webm format
            "video": "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/77/Mouth_of_a_hippopotamus_%28Hippopotamus_amphibius%29.webm/Mouth_of_a_hippopotamus_%28Hippopotamus_amphibius%29.webm.720p.vp9.webm",

            # A youtube video
            "video": "https://www.youtube.com/watch?v=ma67yOdMQfs&ab_channel=RedBullSurfing",  # A youtube video

            # A tiktok video
            "video": "https://www.tiktok.com/@iteleiopu/video/7131608195516108034?is_copy_url=1&is_from_webapp=v1&lang=en",

            # A straight text
            "text": "this is a hello",

            # An image from ndarray
            "image": np.random.randn(200, 300, 3),

            # A video from ndarray
            "video": np.random.randn(8, 200, 500, 3),

        }

        for types, inputs in validation_test_example.items():
            predicted_type = content_routering(inputs)
            assert predicted_type[0] == types, f"We make an incorrect decision on {types} - {inputs}"

    def error_test(self):

        error_test_example = {
            "https://www.bbc.com/reel/video/p0cxqmc3/my-parasite-style-apartment-was-like-a-five-star-hotel": "error",
            # we don't support bbc yet

        }

        for types, inputs in error_test_example.items():
            try:
                predicted_type = content_routering(inputs)
            except TypeError:
                pass
