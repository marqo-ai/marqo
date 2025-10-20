import importlib
import os
import sys
import unittest
from unittest import mock

from inference_orchestrator.config import get_config
from inference_orchestrator.services.inference_cache.caching_inference import (
    CachingInference,
)
from inference_orchestrator.services.triton_inference.triton_inference import (
    TritonInference,
)


class TestInferenceCacheEnvVars(unittest.TestCase):
    def _reset_env_and_config(self):
        """
        Environment variables are cached when the settings module is first imported.
        Config is also cached when the config module is first imported.
        To test different environment variable settings, we need to clear the relevant modules from sys.modules
        """
        importlib.reload(sys.modules["inference_orchestrator.core.settings"])
        importlib.reload(sys.modules["inference_orchestrator.config"])

    def test_api_mode_with_inference_cache(self):
        with mock.patch.dict(
            os.environ,
            {
                "MARQO_INFERENCE_CACHE_SIZE": "10",
            },
            clear=True,
        ):
            self._reset_env_and_config()

            inference = get_config().inference
            self.assertIsInstance(inference, CachingInference)
            self.assertIsInstance(inference.delegate, TritonInference)

    def test_api_mode_without_inference_cache(self):
        with mock.patch.dict(
            os.environ,
            {
                "MARQO_INFERENCE_CACHE_SIZE": "0",
            },
            clear=True,
        ):
            self._reset_env_and_config()

            inference = get_config().inference
            self.assertIsInstance(inference, TritonInference)
