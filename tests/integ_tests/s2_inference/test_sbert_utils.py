import unittest

from marqo.api.exceptions import InternalError
from marqo.s2_inference.sbert_utils import SBERT, Model


class TestSbertLoad(unittest.TestCase):
    def test_sbert_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = "http://example.com/model.pth"
            model = SBERT(model_properties={"url": model_url})
            raise AssertionError
        except InternalError:
            pass

    def test_model_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = "http://example.com/model.pth"
            model = Model(model_properties={"url": model_url})
            raise AssertionError
        except InternalError:
            pass
