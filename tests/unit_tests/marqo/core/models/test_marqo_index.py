import unittest
from typing import Dict, List, Optional

from pydantic.v1 import ValidationError

from marqo.core.models.marqo_index import Field, FieldType, FieldFeature


class TestField(unittest.TestCase):
    """Unit tests for the Field model class."""

    def test_field_creation_minimal(self):
        """Test creating a Field with minimal required parameters."""
        field = Field(
            name="title",
            type=FieldType.Text,
            lexical_field_name=None,
            filter_field_name=None
        )
        self.assertEqual(field.name, "title")
        self.assertEqual(field.type, FieldType.Text)
        self.assertEqual(field.features, [])
        self.assertIsNone(field.lexical_field_name)
        self.assertIsNone(field.filter_field_name)
        self.assertIsNone(field.dependent_fields)
        self.assertIsNone(field.language)
        self.assertIsNone(field.stemming)

    def test_field_creation_with_all_parameters(self):
        """Test creating a Field with all parameters."""
        field = Field(
            name="description",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
            lexical_field_name="description_lexical",
            filter_field_name="description_filter",
            dependent_fields=None,
            language="en",
            stemming="best"
        )
        self.assertEqual(field.name, "description")
        self.assertEqual(field.type, FieldType.Text)
        self.assertEqual(field.features, [FieldFeature.LexicalSearch, FieldFeature.Filter])
        self.assertEqual(field.lexical_field_name, "description_lexical")
        self.assertEqual(field.filter_field_name, "description_filter")
        self.assertIsNone(field.dependent_fields)
        self.assertEqual(field.language, "en")
        self.assertEqual(field.stemming, "best")

    def test_field_creation_multimodal_combination(self):
        """Test creating a MultimodalCombination field with dependent fields."""
        field = Field(
            name="multimodal_field",
            type=FieldType.MultimodalCombination,
            features=[],
            lexical_field_name=None,
            filter_field_name=None,
            dependent_fields={"text_field": 0.7, "image_field": 0.3}
        )
        self.assertEqual(field.name, "multimodal_field")
        self.assertEqual(field.type, FieldType.MultimodalCombination)
        self.assertEqual(field.dependent_fields, {"text_field": 0.7, "image_field": 0.3})

    def test_field_name_validation_invalid_pattern(self):
        """Test that field names must match the required pattern."""
        invalid_names = [
            ("invalid-name!", "contains invalid characters"),
            ("123invalid", "starts with number"),
            ("field with spaces", "contains spaces"),
            ("field@symbol", "contains @ symbol")
        ]

        for invalid_name, description in invalid_names:
            with self.subTest(invalid_name=invalid_name, description=description):
                with self.assertRaises(ValidationError) as cm:
                    Field(
                        name=invalid_name,
                        type=FieldType.Text,
                        lexical_field_name="lexical",
                        filter_field_name="filter"
                    )
                self.assertIn("must match [a-zA-Z_][a-zA-Z0-9_]*", str(cm.exception))

    def test_field_name_validation_reserved_prefix(self):
        """Test that field names cannot start with reserved prefix."""
        with self.assertRaises(ValidationError) as cm:
            Field(
                name="marqo__field",
                type=FieldType.Text,
                lexical_field_name="lexical",
                filter_field_name="filter"
            )
        self.assertIn("must not start with", str(cm.exception))

    def test_field_name_validation_protected_names(self):
        """Test that field names cannot use protected names."""
        protected_names = ["_id", "_tensor_facets", "_highlights", "_score", "_found"]
        for name in protected_names:
            with self.subTest(protected_name=name):
                with self.assertRaises(ValidationError) as cm:
                    Field(
                        name=name,
                        type=FieldType.Text,
                        lexical_field_name="lexical",
                        filter_field_name="filter"
                    )
                self.assertIn("must not be one of", str(cm.exception))

    def test_field_type_feature_compatibility(self):
        """Test all field type and feature combinations for compatibility."""
        # Define all field type and feature combinations with expected results
        # Format: (field_type, field_name, feature, should_be_valid, required_field_names)
        compatibility_matrix = [
            # LexicalSearch feature compatibility
            (FieldType.Text, "text_field", FieldFeature.LexicalSearch, True, {"lexical_field_name": "text_field_lexical"}),
            (FieldType.ArrayText, "array_text_field", FieldFeature.LexicalSearch, True, {"lexical_field_name": "array_text_field_lexical"}),
            (FieldType.CustomVector, "custom_vector_field", FieldFeature.LexicalSearch, True, {"lexical_field_name": "custom_vector_field_lexical"}),
            (FieldType.Bool, "bool_field", FieldFeature.LexicalSearch, False, {}),
            (FieldType.Int, "int_field", FieldFeature.LexicalSearch, False, {}),
            (FieldType.Float, "float_field", FieldFeature.LexicalSearch, False, {}),
            (FieldType.ImagePointer, "image_field", FieldFeature.LexicalSearch, False, {}),
            (FieldType.MultimodalCombination, "multimodal_field", FieldFeature.LexicalSearch, False, {}),

            # ScoreModifier feature compatibility
            (FieldType.Int, "int_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.Long, "long_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.Float, "float_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.Double, "double_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.MapInt, "map_int_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.MapLong, "map_long_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.MapFloat, "map_float_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.MapDouble, "map_double_field", FieldFeature.ScoreModifier, True, {}),
            (FieldType.Text, "text_field", FieldFeature.ScoreModifier, False, {}),
            (FieldType.Bool, "bool_field", FieldFeature.ScoreModifier, False, {}),
            (FieldType.ArrayInt, "array_int_field", FieldFeature.ScoreModifier, False, {}),
            (FieldType.CustomVector, "custom_vector_field", FieldFeature.ScoreModifier, False, {}),
            (FieldType.ImagePointer, "image_field", FieldFeature.ScoreModifier, False, {}),
            (FieldType.MultimodalCombination, "multimodal_field", FieldFeature.ScoreModifier, False, {}),

            # Filter feature compatibility
            (FieldType.Text, "text_field", FieldFeature.Filter, True, {"filter_field_name": "text_field_filter"}),
            (FieldType.Bool, "bool_field", FieldFeature.Filter, True, {"filter_field_name": "bool_field_filter"}),
            (FieldType.Int, "int_field", FieldFeature.Filter, True, {"filter_field_name": "int_field_filter"}),
            (FieldType.Long, "long_field", FieldFeature.Filter, True, {"filter_field_name": "long_field_filter"}),
            (FieldType.Float, "float_field", FieldFeature.Filter, True, {"filter_field_name": "float_field_filter"}),
            (FieldType.Double, "double_field", FieldFeature.Filter, True, {"filter_field_name": "double_field_filter"}),
            (FieldType.ArrayText, "array_text_field", FieldFeature.Filter, True, {"filter_field_name": "array_text_field_filter"}),
            (FieldType.ArrayInt, "array_int_field", FieldFeature.Filter, True, {"filter_field_name": "array_int_field_filter"}),
            (FieldType.ArrayLong, "array_long_field", FieldFeature.Filter, True, {"filter_field_name": "array_long_field_filter"}),
            (FieldType.ArrayFloat, "array_float_field", FieldFeature.Filter, True, {"filter_field_name": "array_float_field_filter"}),
            (FieldType.ArrayDouble, "array_double_field", FieldFeature.Filter, True, {"filter_field_name": "array_double_field_filter"}),
            (FieldType.MapInt, "map_int_field", FieldFeature.Filter, True, {"filter_field_name": "map_int_field_filter"}),
            (FieldType.MapLong, "map_long_field", FieldFeature.Filter, True, {"filter_field_name": "map_long_field_filter"}),
            (FieldType.MapFloat, "map_float_field", FieldFeature.Filter, True, {"filter_field_name": "map_float_field_filter"}),
            (FieldType.MapDouble, "map_double_field", FieldFeature.Filter, True, {"filter_field_name": "map_double_field_filter"}),
            (FieldType.CustomVector, "custom_vector_field", FieldFeature.Filter, True, {"filter_field_name": "custom_vector_field_filter"}),
            (FieldType.VideoPointer, "video_field", FieldFeature.Filter, True, {"filter_field_name": "video_field_filter"}),
            (FieldType.AudioPointer, "audio_field", FieldFeature.Filter, True, {"filter_field_name": "audio_field_filter"}),
            (FieldType.ImagePointer, "image_field", FieldFeature.Filter, False, {}),
            (FieldType.MultimodalCombination, "multimodal_field", FieldFeature.Filter, False, {}),
        ]

        for field_type, field_name, feature, should_be_valid, required_fields in compatibility_matrix:
            with self.subTest(field_type=field_type, feature=feature, expected_valid=should_be_valid):
                # Prepare field arguments
                field_args = {
                    "name": field_name,
                    "type": field_type,
                    "features": [feature],
                    "lexical_field_name": required_fields.get("lexical_field_name"),
                    "filter_field_name": required_fields.get("filter_field_name")
                }

                # Add dependent_fields for MultimodalCombination
                if field_type == FieldType.MultimodalCombination:
                    field_args["dependent_fields"] = {"text": 1.0}

                if should_be_valid:
                    # Should create successfully
                    field = Field(**field_args)
                    self.assertIn(feature, field.features)
                    self.assertEqual(field.type, field_type)
                else:
                    # Should raise ValidationError
                    with self.assertRaises(ValidationError):
                        Field(**field_args)

    def test_language_stemming_field_validation(self):
        """Test that language and stemming fields raise ValidationError when LexicalSearch feature is not present."""

        test_cases = [
            (
                "language_without_lexical_search",
                {"language": "en"},
                "language can only be populated when"
            ),
            (
                "stemming_without_lexical_search",
                {"stemming": "best"},
                "stemming can only be populated when"
            ),
            (
                "both_without_lexical_search",
                {"language": "en", "stemming": "best"},
                "language can only be populated when"  # Language error comes first
            )
        ]

        for case_name, field_config, expected_error in test_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(ValidationError) as cm:
                    Field(
                        name="text_field",
                        type=FieldType.Text,
                        features=[],
                        lexical_field_name=None,
                        filter_field_name=None,
                        language=field_config.get("language"),
                        stemming=field_config.get("stemming")
                    )
                self.assertIn(expected_error, str(cm.exception))

    def test_stemming_value_validation(self):
        """Test that stemming field validates against allowed values."""
        # Test valid stemming values
        valid_values = ["none", "best", "shortest", "multiple"]
        for stemming_value in valid_values:
            with self.subTest(stemming=stemming_value):
                field = Field(
                    name="text_field",
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch],
                    lexical_field_name="text_field_lexical",
                    filter_field_name=None,
                    stemming=stemming_value
                )
                self.assertEqual(field.stemming, stemming_value)

        # Test invalid stemming value
        with self.assertRaises(ValidationError) as cm:
            Field(
                name="text_field",
                type=FieldType.Text,
                features=[FieldFeature.LexicalSearch],
                lexical_field_name="text_field_lexical",
                filter_field_name=None,
                stemming="invalid_value"
            )
        self.assertIn("stemming must be one of", str(cm.exception))

    def test_dependent_fields_validation(self):
        """Test validation for dependent fields in MultimodalCombination type."""
        # Valid: MultimodalCombination with dependent fields
        with self.subTest(test_case="valid_multimodal_with_dependent_fields"):
            field = Field(
                name="multimodal_field",
                type=FieldType.MultimodalCombination,
                features=[],
                lexical_field_name=None,
                filter_field_name=None,
                dependent_fields={"text": 0.6, "image": 0.4}
            )
            self.assertEqual(field.dependent_fields, {"text": 0.6, "image": 0.4})

        # Invalid test cases
        invalid_cases = [
            {
                "name": "text_field",
                "type": FieldType.Text,
                "dependent_fields": {"other": 1.0},
                "expected_error": "dependent_fields must only be defined for fields of type",
                "description": "non_multimodal_with_dependent_fields"
            },
            {
                "name": "multimodal_field",
                "type": FieldType.MultimodalCombination,
                "dependent_fields": None,
                "expected_error": "dependent_fields must be defined",
                "description": "multimodal_without_dependent_fields"
            },
            {
                "name": "multimodal_field",
                "type": FieldType.MultimodalCombination,
                "dependent_fields": {},
                "expected_error": "dependent_fields must be defined",
                "description": "multimodal_with_empty_dependent_fields"
            }
        ]

        for case in invalid_cases:
            with self.subTest(test_case=case["description"]):
                with self.assertRaises(ValidationError) as cm:
                    Field(
                        name=case["name"],
                        type=case["type"],
                        features=[],
                        lexical_field_name=None,
                        filter_field_name=None,
                        dependent_fields=case["dependent_fields"]
                    )
                self.assertIn(case["expected_error"], str(cm.exception))


    def test_required_field_names_validation(self):
        """Test that required field names are present based on features."""
        test_cases = [
            {
                "features": [FieldFeature.LexicalSearch],
                "lexical_field_name": None,
                "filter_field_name": None,
                "expected_error": "lexical_field_name must be populated when",
                "description": "LexicalSearch feature without lexical_field_name"
            },
            {
                "features": [FieldFeature.Filter],
                "lexical_field_name": None,
                "filter_field_name": None,
                "expected_error": "filter_field_name must be populated when",
                "description": "Filter feature without filter_field_name"
            }
        ]

        for test_case in test_cases:
            with self.subTest(description=test_case["description"]):
                with self.assertRaises(ValidationError) as cm:
                    Field(
                        name="text_field",
                        type=FieldType.Text,
                        features=test_case["features"],
                        lexical_field_name=test_case["lexical_field_name"],
                        filter_field_name=test_case["filter_field_name"]
                    )
                self.assertIn(test_case["expected_error"], str(cm.exception))

    def test_field_immutability(self):
        """Test that Field objects are immutable."""
        field = Field(
            name="test_field",
            type=FieldType.Text,
            features=[],
            lexical_field_name=None,
            filter_field_name=None
        )

        # Test that all field attributes are immutable
        immutable_attributes = [
            ("name", "new_name"),
            ("type", FieldType.Int),
            ("features", [FieldFeature.LexicalSearch])
        ]

        for attribute, new_value in immutable_attributes:
            with self.subTest(attribute=attribute):
                with self.assertRaises(TypeError):
                    setattr(field, attribute, new_value)

    def test_multiple_features_combination(self):
        """Test fields with multiple features simultaneously."""
        # Test cases where multiple features are valid for the same field type
        test_cases = [
            {
                "name": "text_field",
                "type": FieldType.Text,
                "features": [FieldFeature.LexicalSearch, FieldFeature.Filter],
                "lexical_field_name": "text_field_lexical",
                "filter_field_name": "text_field_filter",
                "language": "en",
                "stemming": "best",
                "description": "Text with LexicalSearch and Filter"
            },
            {
                "name": "int_field",
                "type": FieldType.Int,
                "features": [FieldFeature.ScoreModifier, FieldFeature.Filter],
                "lexical_field_name": None,
                "filter_field_name": "int_field_filter",
                "language": None,
                "stemming": None,
                "description": "Int with ScoreModifier and Filter"
            },
            {
                "name": "custom_vector_field",
                "type": FieldType.CustomVector,
                "features": [FieldFeature.LexicalSearch, FieldFeature.Filter],
                "lexical_field_name": "custom_vector_lexical",
                "filter_field_name": "custom_vector_filter",
                "language": None,
                "stemming": "multiple",
                "description": "CustomVector with LexicalSearch and Filter"
            }
        ]

        for test_case in test_cases:
            with self.subTest(description=test_case["description"]):
                field = Field(
                    name=test_case["name"],
                    type=test_case["type"],
                    features=test_case["features"],
                    # stemming=test_case["stemming"],
                    lexical_field_name=test_case["lexical_field_name"],
                    filter_field_name=test_case["filter_field_name"],
                    language=test_case["language"]
                )
                self.assertEqual(len(field.features), len(test_case["features"]))
                for feature in test_case["features"]:
                    self.assertIn(feature, field.features)

    def test_field_equality(self):
        """Test field equality comparison."""
        field1 = Field(
            name="test_field",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch],
            lexical_field_name="test_lexical",
            filter_field_name=None,
            language="en",
            stemming="best"
        )

        field2 = Field(
            name="test_field",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch],
            lexical_field_name="test_lexical",
            filter_field_name=None,
            language="en",
            stemming="best"
        )

        field3 = Field(
            name="test_field",
            type=FieldType.Text,
            features=[FieldFeature.LexicalSearch],
            lexical_field_name="test_lexical",
            filter_field_name=None,
            language="es",  # Different language
            stemming = "best"
        )

        self.assertEqual(field1, field2)
        self.assertNotEqual(field1, field3)
