import unittest

# NOTE: circular reference between model_registry & onnx_clip_utils
from marqo.api.exceptions import InternalError
from marqo.s2_inference.onnx_clip_utils import CLIP_ONNX


class TestOnnxClipLoad(unittest.TestCase):
    def test_onnx_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = "http://example.com/model.pth"
            clip = CLIP_ONNX(model_properties={"url": model_url})
            raise AssertionError
        except InternalError:
            pass
