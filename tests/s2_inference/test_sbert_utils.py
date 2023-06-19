import copy
import itertools
import PIL
import requests.exceptions
from marqo.s2_inference import clip_utils, types
import unittest
from unittest import mock
from marqo.s2_inference.sbert_utils import Model, SBERT
from unittest.mock import patch
import pytest
from marqo.errors import InternalError


class TestSbertLoad(unittest.TestCase):
    def test_sbert_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            model = SBERT(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
    
    def test_model_with_no_device(self):
        # Should fail, raising internal error
        try:
            model_url = 'http://example.com/model.pth'
            model = Model(model_properties={'url': model_url})
            raise AssertionError
        except InternalError as e:
            pass
