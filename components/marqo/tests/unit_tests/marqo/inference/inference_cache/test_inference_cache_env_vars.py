import importlib
import os
import sys
import unittest
from unittest import mock

import marqo.tensor_search.api as api

from marqo.core.inference.inference_cache.caching_inference import CachingInference
from marqo.core.inference.inference_client.inference_client import InferenceClient


class TestInferenceCacheEnvVars(unittest.TestCase):
    def test_api_mode_with_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "API",
            "MARQO_API_INFERENCE_CACHE_SIZE": "10",
        }, clear=True):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, CachingInference)
            self.assertIsInstance(inference.delegate, InferenceClient)

    def test_api_mode_without_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "API",
            "MARQO_API_INFERENCE_CACHE_SIZE": "0",
        }, clear=True):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, InferenceClient)
