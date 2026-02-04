import unittest
from unittest.mock import patch

from marqo.core.inference.api.exceptions import UnsupportedModelError
from marqo.core.inference.embedding_models.marqo_model_registry import (
    get_model_properties,
    validate_model_properties,
)


class TestGetModelProperties(unittest.TestCase):
    """Tests for the get_model_properties function."""

    @patch("marqo.core.inference.embedding_models.marqo_model_registry.build_model_properties")
    def test_returns_properties(self, mock_build):
        """Test that model properties are returned correctly."""
        mock_build.return_value = {"dimensions": 384, "type": "hf"}

        result = get_model_properties("test-model", "s3://test-bucket")

        self.assertEqual(result, {"dimensions": 384, "type": "hf"})

    @patch("marqo.core.inference.embedding_models.marqo_model_registry.build_model_properties")
    def test_passes_bucket_to_build_model_properties(self, mock_build):
        """Test that marqo_default_models_s3_bucket is passed correctly."""
        mock_build.return_value = {"dimensions": 384, "type": "hf"}

        get_model_properties("test-model", "s3://my-custom-bucket")

        mock_build.assert_called_once_with("test-model", "s3://my-custom-bucket")

    @patch("marqo.core.inference.embedding_models.marqo_model_registry.build_model_properties")
    def test_uses_default_os_bucket_when_not_specified(self, mock_build):
        """Test that default bucket (os) is used when not specified."""
        mock_build.return_value = {"dimensions": 384, "type": "hf"}

        get_model_properties("test-model")

        mock_build.assert_called_once_with("test-model", "s3://marqo-default-models-os")

    @patch("marqo.core.inference.embedding_models.marqo_model_registry.settings")
    @patch("marqo.core.inference.embedding_models.marqo_model_registry.build_model_properties")
    def test_uses_bucket_from_settings(self, mock_build, mock_settings):
        """Test that bucket from settings is used when settings change."""
        mock_settings.marqo_default_models_s3_bucket = "s3://marqo-default-models-prod"
        mock_build.return_value = {"dimensions": 384, "type": "hf"}

        get_model_properties("test-model", mock_settings.marqo_default_models_s3_bucket)

        mock_build.assert_called_once_with("test-model", "s3://marqo-default-models-prod")

    @patch("marqo.core.inference.embedding_models.marqo_model_registry.build_model_properties")
    def test_raises_unsupported_model_error_for_unknown_model(self, mock_build):
        """Test that unknown models raise UnsupportedModelError."""
        mock_build.side_effect = KeyError("unknown-model")

        with self.assertRaises(UnsupportedModelError):
            get_model_properties("unknown-model", "s3://test-bucket")


class TestValidateModelProperties(unittest.TestCase):
    """Tests for the validate_model_properties function."""

    def test_valid_properties_pass(self):
        """Test that valid properties pass validation."""
        validate_model_properties({"dimensions": 384, "type": "hf"})

    def test_missing_dimensions_raises_error(self):
        """Test that missing 'dimensions' raises ValueError."""
        with self.assertRaises(ValueError):
            validate_model_properties({"type": "hf"})

    def test_missing_type_raises_error(self):
        """Test that missing 'type' raises ValueError."""
        with self.assertRaises(ValueError):
            validate_model_properties({"dimensions": 384})

    def test_empty_dict_raises_error(self):
        """Test that an empty dictionary raises ValueError."""
        with self.assertRaises(ValueError):
            validate_model_properties({})
