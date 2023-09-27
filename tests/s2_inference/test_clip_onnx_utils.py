import copy
import itertools
import PIL
import requests.exceptions
from marqo.s2_inference import clip_utils, types
import unittest
from unittest import mock
import requests
# NOTE: circular reference between model_registry & onnx_clip_utils
import marqo.s2_inference.model_loaders as model_registry
from marqo.s2_inference.onnx_clip_utils import CLIP_ONNX
from marqo.tensor_search.enums import ModelProperties
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
from unittest.mock import patch
import pytest
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
from marqo.tensor_search.models.private_models import S3Auth, S3Location, HfModelLocation
from marqo.s2_inference.configs import ModelCache
from marqo.errors import InternalError


class TestOnnxClipLoad(unittest.TestCase):
    def test_onnx_clip_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            clip = CLIP_ONNX(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
