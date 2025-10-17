import os
from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from inference_orchestrator.core.enum import LogFormat, LogLevel
from inference_orchestrator.core.settings import Settings, get_settings
from inference_orchestrator.schemas.triton_channel_args import TritonChannelArgs


class TestSettings(TestCase):
    def test_default_values(self):
        """Test that Settings initializes with correct default values"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            self.assertEqual(0, settings.marqo_inference_cache_size)
            self.assertEqual("LRU", settings.marqo_inference_cache_type)
            self.assertEqual("http://localhost:8001", settings.marqo_triton_url)
            self.assertEqual(
                "http://localhost:8883", settings.marqo_model_management_container_url
            )
            self.assertEqual([], settings.marqo_models_to_preload)
            self.assertEqual(LogLevel.INFO, settings.marqo_log_level)
            self.assertEqual(LogFormat.PLAIN, settings.marqo_log_format)
            self.assertEqual(30, settings.marqo_metrics_export_interval)
            self.assertIsInstance(settings.channel_args, TritonChannelArgs)

    def test_custom_values_via_environment_variables(self):
        """Test that Settings can be initialized with custom values from environment variables"""
        env_vars = {
            "MARQO_INFERENCE_CACHE_SIZE": "100",
            "MARQO_INFERENCE_CACHE_TYPE": "FIFO",
            "MARQO_TRITON_URL": "http://custom:8001",
            "MARQO_MODEL_MANAGEMENT_CONTAINER_URL": "http://custom:8883",
            "MARQO_MODELS_TO_PRELOAD": '["model1", "model2"]',
            "MARQO_LOG_LEVEL": "DEBUG",
            "MARQO_LOG_FORMAT": "JSON",
            "MARQO_METRICS_EXPORT_INTERVAL": "60",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

            self.assertEqual(100, settings.marqo_inference_cache_size)
            self.assertEqual("FIFO", settings.marqo_inference_cache_type)
            self.assertEqual("http://custom:8001", settings.marqo_triton_url)
            self.assertEqual(
                "http://custom:8883", settings.marqo_model_management_container_url
            )
            self.assertEqual(["model1", "model2"], settings.marqo_models_to_preload)
            self.assertEqual(LogLevel.DEBUG, settings.marqo_log_level)
            self.assertEqual(LogFormat.JSON, settings.marqo_log_format)
            self.assertEqual(60, settings.marqo_metrics_export_interval)

    def test_env_var_aliases(self):
        """Test that environment variable aliases work correctly"""
        test_cases = [
            ("MARQO_INFERENCE_CACHE_SIZE", "marqo_inference_cache_size", "200", 200),
            (
                "MARQO_INFERENCE_CACHE_TYPE",
                "marqo_inference_cache_type",
                "FIFO",
                "FIFO",
            ),
            (
                "MARQO_TRITON_URL",
                "marqo_triton_url",
                "http://test:9000",
                "http://test:9000",
            ),
            (
                "MARQO_MODEL_MANAGEMENT_CONTAINER_URL",
                "marqo_model_management_container_url",
                "http://test:9001",
                "http://test:9001",
            ),
            (
                "MARQO_METRICS_EXPORT_INTERVAL",
                "marqo_metrics_export_interval",
                "45",
                45,
            ),
        ]

        for env_var, attr_name, env_value, expected_value in test_cases:
            with self.subTest(env_var=env_var, expected=expected_value):
                with patch.dict(os.environ, {env_var: env_value}, clear=True):
                    settings = Settings(_env_file=None)
                    actual_value = getattr(settings, attr_name)
                    self.assertEqual(expected_value, actual_value)

    def test_log_level_validator_uppercase_conversion(self):
        """Test that log level validator converts strings to uppercase"""
        test_cases = [
            ("lowercase string", "debug", LogLevel.DEBUG),
            ("uppercase string", "INFO", LogLevel.INFO),
            ("mixed case string", "WaRnInG", LogLevel.WARNING),
            ("error level", "ERROR", LogLevel.ERROR),
        ]

        for msg, input_value, expected_value in test_cases:
            with self.subTest(msg=msg, input=input_value):
                with patch.dict(
                    os.environ, {"MARQO_LOG_LEVEL": input_value}, clear=True
                ):
                    settings = Settings(_env_file=None)
                    self.assertEqual(expected_value, settings.marqo_log_level)

    def test_log_level_validator_none_defaults_to_info(self):
        """Test that log level validator defaults to INFO when not set"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(LogLevel.INFO, settings.marqo_log_level)

    def test_log_format_validator_uppercase_conversion(self):
        """Test that log format validator converts strings to uppercase"""
        test_cases = [
            ("lowercase string", "plain", LogFormat.PLAIN),
            ("uppercase string", "JSON", LogFormat.JSON),
            ("mixed case string", "PlAiN", LogFormat.PLAIN),
            ("json format", "json", LogFormat.JSON),
        ]

        for msg, input_value, expected_value in test_cases:
            with self.subTest(msg=msg, input=input_value):
                with patch.dict(
                    os.environ, {"MARQO_LOG_FORMAT": input_value}, clear=True
                ):
                    settings = Settings(_env_file=None)
                    self.assertEqual(expected_value, settings.marqo_log_format)

    def test_log_format_validator_none_defaults_to_plain(self):
        """Test that log format validator defaults to PLAIN when not set"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(LogFormat.PLAIN, settings.marqo_log_format)

    def test_models_to_preload_string_values(self):
        """Test that models_to_preload accepts string values"""
        models = ["model1", "model2", "model3"]
        with patch.dict(
            os.environ,
            {"MARQO_MODELS_TO_PRELOAD": '["model1", "model2", "model3"]'},
            clear=True,
        ):
            settings = Settings(_env_file=None)
            self.assertEqual(models, settings.marqo_models_to_preload)
            self.assertEqual(3, len(settings.marqo_models_to_preload))

    def test_models_to_preload_dict_values_valid(self):
        """Test that models_to_preload accepts valid dict values with required keys"""
        models_json = '[{"model": "custom-model-1", "modelProperties": {"type": "clip"}}, {"model": "custom-model-2", "modelProperties": {"type": "bert"}}]'
        with patch.dict(
            os.environ, {"MARQO_MODELS_TO_PRELOAD": models_json}, clear=True
        ):
            settings = Settings(_env_file=None)
            self.assertEqual(2, len(settings.marqo_models_to_preload))
            self.assertIn("model", settings.marqo_models_to_preload[0])
            self.assertIn("modelProperties", settings.marqo_models_to_preload[0])

    def test_models_to_preload_dict_missing_model_key(self):
        """Test that models_to_preload raises error when dict is missing 'model' key"""
        models_json = '[{"modelProperties": {"type": "clip"}}]'

        with self.assertRaises(ValidationError) as context:
            with patch.dict(
                os.environ, {"MARQO_MODELS_TO_PRELOAD": models_json}, clear=True
            ):
                Settings(_env_file=None)

        self.assertIn("model", str(context.exception).lower())

    def test_models_to_preload_dict_missing_modelProperties_key(self):
        """Test that models_to_preload raises error when dict is missing 'modelProperties' key"""
        models_json = '[{"model": "custom-model"}]'

        with self.assertRaises(ValidationError) as context:
            with patch.dict(
                os.environ, {"MARQO_MODELS_TO_PRELOAD": models_json}, clear=True
            ):
                Settings(_env_file=None)

        self.assertIn("modelproperties", str(context.exception).lower())

    def test_models_to_preload_mixed_string_and_dict(self):
        """Test that models_to_preload accepts mixed string and dict values"""
        models_json = '["model1", {"model": "custom-model", "modelProperties": {"type": "clip"}}, "model2"]'
        with patch.dict(
            os.environ, {"MARQO_MODELS_TO_PRELOAD": models_json}, clear=True
        ):
            settings = Settings(_env_file=None)
            self.assertEqual(3, len(settings.marqo_models_to_preload))
            self.assertEqual("model1", settings.marqo_models_to_preload[0])
            self.assertIsInstance(settings.marqo_models_to_preload[1], dict)
            self.assertEqual("model2", settings.marqo_models_to_preload[2])

    def test_models_to_preload_empty_list(self):
        """Test that models_to_preload accepts empty list"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual([], settings.marqo_models_to_preload)
            self.assertEqual(0, len(settings.marqo_models_to_preload))

    def test_metrics_export_interval_validation(self):
        """Test that metrics_export_interval validates non-negative values"""
        valid_cases = [
            ("zero", "0", 0),
            ("positive", "30", 30),
            ("large positive", "1000", 1000),
        ]

        for msg, env_value, expected_value in valid_cases:
            with self.subTest(msg=msg, value=expected_value):
                with patch.dict(
                    os.environ, {"MARQO_METRICS_EXPORT_INTERVAL": env_value}, clear=True
                ):
                    settings = Settings(_env_file=None)
                    self.assertEqual(
                        expected_value, settings.marqo_metrics_export_interval
                    )

    def test_metrics_export_interval_negative_value(self):
        """Test that metrics_export_interval rejects negative values"""
        with self.assertRaises(ValidationError) as context:
            with patch.dict(
                os.environ, {"MARQO_METRICS_EXPORT_INTERVAL": "-1"}, clear=True
            ):
                Settings(_env_file=None)

        self.assertIn("greater than or equal to 0", str(context.exception).lower())

    def test_channel_args_default(self):
        """Test that channel_args uses default TritonChannelArgs"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertIsInstance(settings.channel_args, TritonChannelArgs)
            self.assertEqual(20_000, settings.channel_args.grpc_keep_alive_time_ms)

    def test_channel_args_custom_via_json(self):
        """Test that channel_args can be customized via JSON environment variable"""
        channel_args_json = '{"grpc_keep_alive_time_ms": 30000}'
        with patch.dict(
            os.environ, {"MARQO_TRITON_CHANNEL_ARGS": channel_args_json}, clear=True
        ):
            settings = Settings(_env_file=None)
            self.assertEqual(30_000, settings.channel_args.grpc_keep_alive_time_ms)

    def test_settings_immutability(self):
        """Test that Settings is frozen and cannot be modified after creation"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            with self.assertRaises(ValidationError):
                settings.marqo_inference_cache_size = 999

    def test_settings_frozen_multiple_fields(self):
        """Test that all Settings fields are frozen and cannot be modified"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            # Test that various fields cannot be modified
            test_cases = [
                ("marqo_inference_cache_size", 100),
                ("marqo_inference_cache_type", "FIFO"),
                ("marqo_triton_url", "http://modified:8001"),
                ("marqo_log_level", LogLevel.DEBUG),
                ("marqo_log_format", LogFormat.JSON),
                ("marqo_metrics_export_interval", 60),
            ]

            for field_name, new_value in test_cases:
                with self.subTest(field=field_name):
                    with self.assertRaises(ValidationError) as context:
                        setattr(settings, field_name, new_value)
                    self.assertIn("frozen", str(context.exception).lower())

    def test_get_settings_returns_singleton(self):
        """Test that get_settings returns the same instance"""
        settings1 = get_settings()
        settings2 = get_settings()

        self.assertIs(settings1, settings2)
        self.assertIsInstance(settings1, Settings)

    def test_populate_by_name(self):
        """Test that settings can be populated by field name or alias"""
        # Test using alias via environment variable
        with patch.dict(os.environ, {"MARQO_INFERENCE_CACHE_SIZE": "200"}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(200, settings.marqo_inference_cache_size)
