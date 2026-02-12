import unittest
from pydantic import ValidationError
from unittest.mock import patch

from marqo.settings.settings import (
    Settings,
    get_settings,
)


class TestSettings(unittest.TestCase):
    """Tests for the Settings pydantic-settings class."""

    def test_default_bucket_value(self):
        """Test that the default bucket is set to the OS bucket."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            self.assertEqual(settings.marqo_default_models_s3_bucket, "s3://marqo-default-models-os")

    def test_bucket_from_env_var(self):
        """Test setting bucket via the MARQO_DEFAULT_MODELS_S3_BUCKET environment variable."""
        test_cases = [
            "s3://marqo-default-models-os",
            "s3://marqo-default-models-staging",
            "s3://marqo-default-models-preprod",
            "s3://marqo-default-models-production",
            "s3://my-custom-bucket",
        ]
        for bucket in test_cases:
            with self.subTest(bucket=bucket):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": bucket}, clear=True):
                    settings = Settings()
                    self.assertEqual(settings.marqo_default_models_s3_bucket, bucket)

    def test_bucket_adds_s3_prefix_when_missing(self):
        """Test that the validator adds the s3:// prefix if it's missing."""
        test_cases = [
            ("marqo-default-models-os", "s3://marqo-default-models-os"),
            ("my-custom-bucket", "s3://my-custom-bucket"),
        ]
        for env_value, expected in test_cases:
            with self.subTest(env_value=env_value):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": env_value}, clear=True):
                    settings = Settings()
                    self.assertEqual(settings.marqo_default_models_s3_bucket, expected)

    def test_bucket_removes_trailing_slash(self):
        """Test that the validator removes trailing slashes without corrupting the s3:// prefix."""
        test_cases = [
            ("s3://marqo-default-models-os/", "s3://marqo-default-models-os"),
            ("s3://my-bucket///", "s3://my-bucket"),
        ]
        for env_value, expected in test_cases:
            with self.subTest(env_value=env_value):
                with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": env_value}, clear=True):
                    settings = Settings()
                    self.assertEqual(settings.marqo_default_models_s3_bucket, expected)

    def test_bucket_adds_prefix_and_removes_trailing_slash(self):
        """Test that the validator both adds s3:// prefix and removes trailing slashes."""
        with patch.dict("os.environ", {"MARQO_DEFAULT_MODELS_S3_BUCKET": "my-bucket/"}, clear=True):
            settings = Settings()
            self.assertEqual(settings.marqo_default_models_s3_bucket, "s3://my-bucket")

    def test_settings_is_frozen(self):
        """Test that the Settings instance is immutable (frozen)."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            with self.assertRaises(ValidationError):
                settings.marqo_default_models_s3_bucket = "s3://some-other-bucket"

    def test_settings_ignores_extra_fields(self):
        """Test that Settings ignores extra environment variables."""
        with patch.dict("os.environ", {"SOME_OTHER_VAR": "some_value"}, clear=True):
            # Should not raise an error
            settings = Settings()
            self.assertEqual(settings.marqo_default_models_s3_bucket, "s3://marqo-default-models-os")


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

    def test_get_settings_has_bucket(self):
        """Test that the returned settings have a bucket value."""
        settings = get_settings()
        self.assertIsInstance(settings.marqo_default_models_s3_bucket, str)
        self.assertTrue(len(settings.marqo_default_models_s3_bucket) > 0)


