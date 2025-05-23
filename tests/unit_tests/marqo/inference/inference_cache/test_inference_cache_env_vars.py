import importlib
import os
import sys
import unittest
from unittest import mock

import marqo.tensor_search.api as api
import marqo.inference.native_inference.remote.server.inference_api as inference_api

from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.inference.native_inference.remote.client.inference_client import NativeInferenceClient


class TestInferenceCacheEnvVars(unittest.TestCase):

    def test_combined_mode_with_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "COMBINED",
            "MARQO_INFERENCE_SERVER_CACHE_SIZE": "10",
        }):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, CachingInference)
            self.assertIsInstance(inference.delegate, NativeInferenceLocal)
            self.assertTrue(inference.inference_cache.is_enabled())

    def test_combined_mode_without_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "COMBINED",
            "MARQO_INFERENCE_SERVER_CACHE_SIZE": "0",
        }):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, NativeInferenceLocal)

    def test_api_mode_with_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "API",
            "MARQO_INFERENCE_CLIENT_CACHE_SIZE": "10",
        }):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, CachingInference)
            self.assertIsInstance(inference.delegate, NativeInferenceClient)
            self.assertTrue(inference.inference_cache.is_enabled())

    def test_api_mode_without_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "API",
            "MARQO_INFERENCE_CLIENT_CACHE_SIZE": "0",
        }):
            importlib.reload(sys.modules['marqo.tensor_search.api'])

            inference = api.get_config().inference
            self.assertIsInstance(inference, NativeInferenceClient)

    def test_inference_mode_with_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "INFERENCE",
            "MARQO_INFERENCE_SERVER_CACHE_SIZE": "10",
        }):
            importlib.reload(sys.modules['marqo.inference.native_inference.remote.server.inference_api'])

            inference = inference_api.get_config().local_inference
            self.assertIsInstance(inference, CachingInference)
            self.assertIsInstance(inference.delegate, NativeInferenceLocal)
            self.assertTrue(inference.inference_cache.is_enabled())

    def test_inference_mode_without_inference_cache(self):
        with mock.patch.dict(os.environ, {
            "MARQO_MODE": "INFERENCE",
            "MARQO_INFERENCE_SERVER_CACHE_SIZE": "0",
        }):
            importlib.reload(sys.modules['marqo.inference.native_inference.remote.server.inference_api'])

            inference = inference_api.get_config().local_inference
            self.assertIsInstance(inference, NativeInferenceLocal)
