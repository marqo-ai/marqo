import importlib
import json
import os
import sys
import unittest
from unittest import mock

from inference_orchestrator import on_start_script
from inference_orchestrator.config import Config
from inference_orchestrator.errors.common_errors import (
    EnvironmentVariableParsingError,
)
from inference_orchestrator.schemas.api import Inference


class TestOnStartScript(unittest.TestCase):
    def setUp(self):
        self.mock_inference = mock.MagicMock(spec=Inference)
        self.mock_config = mock.MagicMock(spec=Config, inference=self.mock_inference)

    def _help_reload_envs(self):
        importlib.reload(sys.modules["inference_orchestrator.core.settings"])
        importlib.reload(sys.modules["inference_orchestrator.on_start_script"])

    def test_preload_registry_models(self):
        environ_expected_models = [
            ({"MARQO_MODELS_TO_PRELOAD": "[]"}, []),
            ({}, []),
            ({"MARQO_MODELS_TO_PRELOAD": '["hf/e5-base-v2"]'}, ["hf/e5-base-v2"]),
            (
                {"MARQO_MODELS_TO_PRELOAD": '["Marqo/marqo-fashionSigLIP"]'},
                ["Marqo/marqo-fashionSigLIP"],
            ),
        ]
        for mock_environ, expected in environ_expected_models:
            self.mock_inference.reset_mock()
            with mock.patch.dict(os.environ, mock_environ, clear=True):
                self._help_reload_envs()
                model_caching_script = on_start_script.CacheModels(self.mock_config)
                model_caching_script.run()
                loaded_models = {
                    args[0].embedding_model_config.model_name
                    for args, _ in self.mock_inference.vectorise.call_args_list
                }
                self.assertEqual(set(expected), loaded_models)

    def test_preload_custom_models(self):
        dummy_model_properties = {
            # A dummy model property for testing, not valid
            "name": "hf-hub:my-custom-model",
            "dimensions": 512,
            "type": "open_clip",
            "tritonImageEncoder": {"test": "dummy"},
        }
        model_name = "my-custom-model"

        models_to_preload = [
            {"model": model_name, "modelProperties": dummy_model_properties}
        ]

        with mock.patch.dict(
            os.environ,
            {"MARQO_MODELS_TO_PRELOAD": json.dumps(models_to_preload)},
            clear=True,
        ):
            self.mock_inference.reset_mock()
            self._help_reload_envs()
            model_caching_script = on_start_script.CacheModels(self.mock_config)
            model_caching_script.run()
            loaded_models = (
                self.mock_inference.vectorise.call_args_list[0]
                .args[0]
                .embedding_model_config.model_name
            )
            loaded_model_properties = (
                self.mock_inference.vectorise.call_args_list[0]
                .args[0]
                .embedding_model_config.model_properties
            )
            self.assertEqual("my-custom-model", loaded_models)
            self.assertEqual(dummy_model_properties, loaded_model_properties)

    def test_preload_invalid_format_model(self):
        """
        Invalid formats will raise EnvironmentVariableParsingError before reaching the model loading stage. More
        specific model loading errors are tested in test_settings.py as the environment variable parsing is now
        centralized there.
        """
        environ_expected_models = [
            ({"MARQO_MODELS_TO_PRELOAD": "invalid format"}, []),
        ]
        for mock_environ, expected in environ_expected_models:
            self.mock_inference.reset_mock()
            with self.assertRaises(EnvironmentVariableParsingError):
                with mock.patch.dict(os.environ, mock_environ, clear=True):
                    self._help_reload_envs()
                    model_caching_script = on_start_script.CacheModels(self.mock_config)
                    model_caching_script.run()
                    loaded_models = {
                        args[0].embedding_model_config.model_name
                        for args, _ in self.mock_inference.vectorise.call_args_list
                    }
                    self.assertEqual(set(expected), loaded_models)
