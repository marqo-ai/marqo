import json
from unittest import TestCase
from unittest.mock import patch

from model_management.core.enum import LogFormat, LogLevel
from model_management.core.settings import Settings
from model_management.schemas.triton_model_properties import TritonModelProperties
from pydantic import ValidationError
from pydantic_settings import SettingsError


def _help_get_settings_without_dota_env():
    """
    Helper function to get Settings without loading from .env file.
    This is useful for tests to avoid interference from existing .env files.
    """
    return Settings(_env_file=None)


class TestSettings(TestCase):
    """
    A test class for the Settings class in model_management.core.settings.

    Note that the test could interact with your .env file if it exists as Pydantic's BaseSettings will automatically
    load environment variables from a .env file. To avoid this, use the helper function
    _help_get_settings_without_dota_env() which sets _env_file=None.
    """

    def test_default_values(self):
        """Test that default values are set correctly when no environment variables are provided."""
        with patch("os.environ", {}):
            settings = _help_get_settings_without_dota_env()
            self.assertEqual("http://localhost:8000", settings.marqo_triton_rest_url)
            # marqo_model_cache_path has a dynamic default based on home directory
            self.assertIsNotNone(settings.marqo_model_cache_path)
            self.assertEqual(LogLevel.INFO, settings.marqo_log_level)
            self.assertEqual(LogFormat.PLAIN, settings.marqo_log_format)
            self.assertEqual([], settings.marqo_models_to_preload)

    def test_customised_values(self):
        """Test that custom environment variable values are parsed correctly."""
        model_to_preload = [
            {
                "maxBatchSize": 8,
                "name": "marqo-fashionSigLIP-image-encoder",
                "sources": [
                    "s3://opensource-li-backup/triton_models/marqo-fashionSigLIP-image-encoder/1/model.onnx"
                ],
                "input": [
                    {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
                ],
                "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
            }
        ]

        custom_values = {
            "MARQO_TRITON_REST_URL": "http://custom-triton:8000",
            "MARQO_MODEL_CACHE_PATH": "/custom/models",
            "MARQO_LOG_LEVEL": "DEBUG",
            "MARQO_LOG_FORMAT": "JSON",
            "MARQO_MODELS_TO_PRELOAD": json.dumps(model_to_preload),
        }
        with patch("os.environ", custom_values):
            settings = (
                _help_get_settings_without_dota_env()
            )  # Avoid loading from .env file during tests
            self.assertEqual(
                custom_values["MARQO_TRITON_REST_URL"], settings.marqo_triton_rest_url
            )
            self.assertEqual(
                custom_values["MARQO_MODEL_CACHE_PATH"], settings.marqo_model_cache_path
            )
            self.assertEqual(custom_values["MARQO_LOG_LEVEL"], settings.marqo_log_level)
            self.assertEqual(
                custom_values["MARQO_LOG_FORMAT"], settings.marqo_log_format
            )
            self.assertEqual(1, len(settings.marqo_models_to_preload))
            self.assertEqual(
                TritonModelProperties(**model_to_preload[0]),
                settings.marqo_models_to_preload[0],
            )

    def test_incorrect_models_to_preload(self):
        ill_model_to_preload_test_cases = [
            ("test", "not a json array"),
            ("{}", "not a json array"),
            (
                '[{"maxBatchSize": 8, "name": "marqo-fashionSigLIP-image-encoder"}]',
                "incomplete model properties",
            ),
        ]
        for ill_value, msg in ill_model_to_preload_test_cases:
            with self.subTest(msg):
                with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": ill_value}):
                    with self.assertRaises((SettingsError, ValidationError)):
                        _ = _help_get_settings_without_dota_env()

    def test_marqo_models_to_preload_max_length_constraint(self):
        """Test that marqo_models_to_preload enforces max_length=3 constraint."""
        model = {
            "maxBatchSize": 8,
            "name": "test-model",
            "sources": ["s3://test/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        }

        test_cases = [
            (1, "single model should pass"),
            (2, "two models should pass"),
            (3, "three models should pass"),
            (4, "four models should fail"),
            (5, "five models should fail"),
        ]

        for num_models, msg in test_cases:
            with self.subTest(msg=msg):
                models = [
                    dict(model, name=f"test-model-{i}") for i in range(num_models)
                ]
                env_value = json.dumps(models)

                with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": env_value}):
                    if num_models <= 3:
                        settings = _help_get_settings_without_dota_env()
                        self.assertEqual(
                            num_models, len(settings.marqo_models_to_preload)
                        )
                    else:
                        with self.assertRaises((SettingsError, ValidationError)):
                            _ = _help_get_settings_without_dota_env()

    def test_marqo_models_to_preload_edge_cases(self):
        """Test edge cases for marqo_models_to_preload JSON parsing."""
        test_cases = [
            (None, [], "None string should return empty list"),
            ("[]", [], "empty array should return empty list"),
        ]

        for env_value, expected, msg in test_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": env_value}):
                    settings = _help_get_settings_without_dota_env()
                    self.assertEqual(expected, settings.marqo_models_to_preload)

    def test_marqo_models_to_preload_with_json_env_variable(self):
        """Test that marqo_models_to_preload works with JSON environment variable."""
        model = {
            "maxBatchSize": 8,
            "name": "test-model",
            "sources": ["s3://test/model.onnx"],
            "input": [
                {"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}
            ],
            "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        }

        # Test with JSON string in environment variable (the intended way)
        with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": json.dumps([model])}):
            settings = _help_get_settings_without_dota_env()
            self.assertEqual(1, len(settings.marqo_models_to_preload))
            self.assertEqual(
                TritonModelProperties(**model), settings.marqo_models_to_preload[0]
            )

    def test_log_level_validation(self):
        """Test log level validation with different cases and values."""
        test_cases = [
            (
                "debug",
                LogLevel.DEBUG,
                "lowercase debug should be converted to uppercase",
            ),
            ("INFO", LogLevel.INFO, "uppercase INFO should remain uppercase"),
            (
                "Warning",
                LogLevel.WARNING,
                "mixed case Warning should be converted to uppercase",
            ),
            (
                "error",
                LogLevel.ERROR,
                "lowercase error should be converted to uppercase",
            ),
            (None, LogLevel.INFO, "None should default to INFO"),
        ]

        for env_value, expected, msg in test_cases:
            with self.subTest(msg=msg):
                env_dict = (
                    {"MARQO_LOG_LEVEL": env_value} if env_value is not None else {}
                )
                with patch("os.environ", env_dict):
                    settings = _help_get_settings_without_dota_env()
                    self.assertEqual(expected, settings.marqo_log_level)

    def test_log_format_validation(self):
        """Test log format validation with different cases and values."""
        test_cases = [
            (
                "plain",
                LogFormat.PLAIN,
                "lowercase plain should be converted to uppercase",
            ),
            ("JSON", LogFormat.JSON, "uppercase JSON should remain uppercase"),
            (
                "Plain",
                LogFormat.PLAIN,
                "mixed case Plain should be converted to uppercase",
            ),
            ("json", LogFormat.JSON, "lowercase json should be converted to uppercase"),
            (None, LogFormat.PLAIN, "None should default to PLAIN"),
        ]

        for env_value, expected, msg in test_cases:
            with self.subTest(msg=msg):
                env_dict = (
                    {"MARQO_LOG_FORMAT": env_value} if env_value is not None else {}
                )
                with patch("os.environ", env_dict):
                    settings = _help_get_settings_without_dota_env()
                    self.assertEqual(expected, settings.marqo_log_format)

    def test_invalid_log_level_and_format(self):
        """Test that invalid log levels and formats raise ValidationError."""
        invalid_cases = [
            ({"MARQO_LOG_LEVEL": "INVALID"}, "invalid log level"),
            ({"MARQO_LOG_FORMAT": "INVALID"}, "invalid log format"),
            ({"MARQO_LOG_LEVEL": "TRACE"}, "unsupported log level"),
            ({"MARQO_LOG_FORMAT": "XML"}, "unsupported log format"),
        ]

        for env_dict, msg in invalid_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", env_dict):
                    with self.assertRaises(ValidationError):
                        _help_get_settings_without_dota_env()

    def test_get_settings_environment_variables_parsing_error(self):
        """Test that get_settings raises EnvironmentVariablesParsingError for invalid environment variables."""
        invalid_env_cases = [
            (
                {"MARQO_MODELS_TO_PRELOAD": "invalid json"},
                "invalid JSON in MARQO_MODELS_TO_PRELOAD",
            ),
            ({"MARQO_LOG_LEVEL": "INVALID_LEVEL"}, "invalid MARQO_LOG_LEVEL"),
            ({"MARQO_LOG_FORMAT": "INVALID_FORMAT"}, "invalid MARQO_LOG_FORMAT"),
        ]

        for env_dict, msg in invalid_env_cases:
            with self.subTest(msg=msg):
                # Clear cache to ensure fresh settings
                with patch("os.environ", env_dict):
                    with self.assertRaises((SettingsError, ValidationError)):
                        _ = _help_get_settings_without_dota_env()

    def test_marqo_models_to_preload_invalid_json_strings(self):
        """Test that marqo_models_to_preload rejects invalid JSON strings."""
        invalid_cases = [
            ("123", "integer JSON should be rejected"),
            ('{"key": "value"}', "object JSON should be rejected"),
            ("true", "boolean JSON should be rejected"),
            ("invalid json", "malformed JSON should be rejected"),
            (" ", "whitespace string should be rejected"),
            ("[", "Incomplete JSON array should be rejected"),
        ]

        for invalid_value, msg in invalid_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_MODELS_TO_PRELOAD": invalid_value}):
                    with self.assertRaises((SettingsError, ValidationError)):
                        _help_get_settings_without_dota_env()

    def test_triton_url_custom_values(self):
        """Test marqo_triton_rest_url accepts various valid URL formats."""
        url_cases = [
            ("http://localhost:8000", "default localhost URL"),
            ("https://triton.example.com:8000", "HTTPS URL with domain"),
            ("http://192.168.1.100:9000", "IP address with custom port"),
            ("http://triton-service", "service name without port"),
        ]

        for url, msg in url_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_TRITON_REST_URL": url}):
                    settings = _help_get_settings_without_dota_env()
                    self.assertEqual(url, settings.marqo_triton_rest_url)

    def test_marqo_model_cache_path_custom_values(self):
        """Test marqo_model_cache_path accepts various path formats."""
        path_cases = [
            ("./cache/models", "relative path with dot"),
            ("/tmp/models", "absolute path"),
            ("~/models", "home directory path"),
            ("models", "simple relative path"),
        ]

        for path, msg in path_cases:
            with self.subTest(msg=msg):
                with patch("os.environ", {"MARQO_MODEL_CACHE_PATH": path}):
                    settings = _help_get_settings_without_dota_env()
                    self.assertEqual(path, settings.marqo_model_cache_path)
