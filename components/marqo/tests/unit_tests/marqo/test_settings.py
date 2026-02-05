import importlib
import unittest
from pydantic import ValidationError
from unittest.mock import patch

from marqo.api.exceptions import EnvVarError
from marqo.settings import settings as settings_module
from marqo.settings.settings import (
    MarqoDefaultModelsBucket,
    Settings,
    get_settings,
)


class TestSettings(unittest.TestCase):
    """Tests for the Settings pydantic-settings class."""

    def test_default_bucket_value(self):
        """Test that the default bucket is set to 'os'."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            self.assertEqual(settings.marqo_default_models_s3_bucket, MarqoDefaultModelsBucket.os)

    def test_bucket_from_shortcut_name(self):
        """Test setting bucket via shortcut names (os, staging, preprod, prod)."""
        test_cases = [
            ("os", MarqoDefaultModelsBucket.os),
            ("staging", MarqoDefaultModelsBucket.staging),
            ("preprod", MarqoDefaultModelsBucket.preprod),
            ("prod", MarqoDefaultModelsBucket.prod),
        ]
        for shortcut, expected_bucket in test_cases:
            with self.subTest(shortcut=shortcut):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": shortcut}, clear=True):
                    settings = Settings()
                    self.assertEqual(settings.marqo_default_models_s3_bucket, expected_bucket)

    def test_bucket_from_full_s3_url(self):
        """Test setting bucket via full S3 URLs."""
        test_cases = [
            ("s3://marqo-default-models-os", MarqoDefaultModelsBucket.os),
            ("s3://marqo-default-models-staging", MarqoDefaultModelsBucket.staging),
            ("s3://marqo-default-models-preprod", MarqoDefaultModelsBucket.preprod),
            ("s3://marqo-default-models-prod", MarqoDefaultModelsBucket.prod),
        ]
        for s3_url, expected_bucket in test_cases:
            with self.subTest(s3_url=s3_url):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": s3_url}, clear=True):
                    settings = Settings()
                    self.assertEqual(settings.marqo_default_models_s3_bucket, expected_bucket)

    def test_invalid_bucket_value_raises_error(self):
        """Test that invalid bucket values raise a ValidationError."""
        invalid_values = ["invalid_bucket", "s3://some-other-bucket", "dev", "production"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": invalid_value}, clear=True):
                    with self.assertRaises(ValidationError) as context:
                        Settings()
                    self.assertIn("MARQO_DEFAULT_MODELS_S3_BUCKET", str(context.exception))

    def test_settings_is_frozen(self):
        """Test that the Settings instance is immutable (frozen)."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            with self.assertRaises(ValidationError):
                settings.marqo_default_models_s3_bucket = MarqoDefaultModelsBucket.prod

    def test_settings_ignores_extra_fields(self):
        """Test that Settings ignores extra environment variables."""
        with patch.dict("os.environ", {"SOME_OTHER_VAR": "some_value"}, clear=True):
            # Should not raise an error
            settings = Settings()
            self.assertEqual(settings.marqo_default_models_s3_bucket, MarqoDefaultModelsBucket.os)


class TestGetSettings(unittest.TestCase):
    """Tests for the get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        self.assertIsInstance(settings, Settings)

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same module-level instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        self.assertIs(settings1, settings2)

    def test_get_settings_has_valid_bucket(self):
        """Test that the returned settings have a valid bucket value."""
        settings = get_settings()
        self.assertIn(settings.marqo_default_models_s3_bucket, list(MarqoDefaultModelsBucket))


class TestSettingsModuleLoadError(unittest.TestCase):
    """Tests for the module-level exception handling during settings initialization."""

    def test_invalid_settings_raises_env_var_error_on_module_load(self):
        """Test that ValidationError/SettingsError is converted to EnvVarError at module load.

        This tests lines 40-41 of settings.py where module-level exception handling
        converts pydantic errors to EnvVarError.
        """
        with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": "invalid_bucket"}, clear=True):
            with self.assertRaises(EnvVarError) as context:
                importlib.reload(settings_module)
            self.assertIn("Error parsing environment variables", str(context.exception))