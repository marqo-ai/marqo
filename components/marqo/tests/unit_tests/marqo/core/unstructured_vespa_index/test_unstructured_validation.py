import unittest
from unittest.mock import MagicMock

from marqo.api.exceptions import InvalidArgError, InvalidFieldNameError
from marqo.core.models import marqo_index
import marqo.core.unstructured_vespa_index.unstructured_validation as validation
from marqo.exceptions import InvalidArgumentError


class TestValidateMappingsObject(unittest.TestCase):
    """Unit tests for the validate_mappings_object function."""

    def test_validate_mappings_object_valid_cases(self):
        """Test validation of valid mapping configurations."""
        test_cases = [
            {
                "description": "multimodal combination",
                "mapping": {
                    "combined_field": {
                        "type": "multimodal_combination",
                        "weights": {"text": 0.7, "image": 0.3}
                    }
                }
            },
            {
                "description": "custom vector",
                "mapping": {
                    "vector_field": {"type": "custom_vector"}
                }
            },
            {
                "description": "text field with language",
                "mapping": {
                    "text_field": {"type": "text_field", "language": "es"}
                }
            },
            {
                "description": "mixed field types",
                "mapping": {
                    "multimodal": {"type": "multimodal_combination", "weights": {"text": 1.0}},
                    "vector": {"type": "custom_vector"},
                    "text": {"type": "text_field", "language": "en"}
                }
            },
            {
                "description": "empty multimodal weights",
                "mapping": {
                    "field": {"type": "multimodal_combination", "weights": {}}
                }
            }
        ]

        for case in test_cases:
            with self.subTest(description=case["description"]):
                validation.validate_mappings_object_format(case["mapping"])

    def test_validate_mappings_object_invalid_cases(self):
        """Test validation errors for invalid mapping configurations."""
        test_cases = [
            {
                "description": "multimodal missing weights",
                "mapping": {"field": {"type": "multimodal_combination"}},
                "expected_error": "'weights' is a required property",
                "exception_type": InvalidArgError
            },
            {
                "description": "multimodal non-numeric weight",
                "mapping": {"field": {"type": "multimodal_combination", "weights": {"text": "invalid"}}},
                "expected_error": "is not of type 'number'",
                "exception_type": InvalidArgError
            },
            {
                "description": "custom vector extra properties",
                "mapping": {"field": {"type": "custom_vector", "extra": "not_allowed"}},
                "expected_error": "Additional properties are not allowed",
                "exception_type": InvalidArgError
            },
            {
                "description": "text field missing language and stemming",
                "mapping": {"field": {"type": "text_field"}},
                "expected_error": "not valid under any of the given schemas",
                "exception_type": InvalidArgumentError
            },
            {
                "description": "text field empty language",
                "mapping": {"field": {"type": "text_field", "language": ""}},
                "expected_error": "'' should be non-empty",
                "exception_type": InvalidArgumentError
            },
            {
                "description": "unknown mapping type",
                "mapping": {"field": {"type": "unknown_type"}},
                "expected_error": "'unknown_type' is not one of",
                "exception_type": InvalidArgError
            }
        ]

        for case in test_cases:
            with self.subTest(description=case["description"]):
                with self.assertRaises(case["exception_type"]) as cm:
                    validation.validate_mappings_object_format(case["mapping"])
                self.assertIn(case["expected_error"], str(cm.exception))
