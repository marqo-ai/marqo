"""Unit tests for RecencyParameters model."""
import unittest
from pydantic.v1 import ValidationError

from marqo.tensor_search.models.recency_parameters import RecencyParameters


class TestRecencyParameters(unittest.TestCase):
    """Tests for RecencyParameters model."""

    def test_recency_field_validation(self):
        """Test recency_field validation."""
        # Valid cases
        valid_cases = [
            ("minimal_parameters", {"recency_field": "created_at"}, {
                "recency_field": "created_at",
                "decay_function": "exponential",
                "scale": "7d",
                "offset": "0d",
                "decay_to": 0.5,
                "apply_in_ranking_phase": "all"
            }),
            ("all_parameters", {
                "recency_field": "updated_at",
                "decay_function": "linear",
                "scale": "14d",
                "offset": "1d",
                "decay_to": 0.3,
                "apply_in_ranking_phase": "only-global"
            }, {
                "recency_field": "updated_at",
                "decay_function": "linear",
                "scale": "14d",
                "offset": "1d",
                "decay_to": 0.3,
                "apply_in_ranking_phase": "only-global"
            }),
            ("whitespace_trimmed", {"recency_field": "  created_at  "}, {
                "recency_field": "created_at"
            }),
        ]

        for test_name, input_params, expected_values in valid_cases:
            with self.subTest(test_name):
                params = RecencyParameters(**input_params)
                for key, expected_value in expected_values.items():
                    self.assertEqual(getattr(params, key), expected_value)

        # Invalid cases
        invalid_cases = [
            ("required", {}, ["recency_field", "recencyField"]),
            ("empty_string", {"recency_field": ""}, ["recency_field", "recencyField"]),
            ("whitespace_only", {"recency_field": "   "}, ["recency_field", "recencyField"]),
        ]

        for test_name, input_params, expected_fields in invalid_cases:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(**input_params)
                errors = exc_info.exception.errors()
                self.assertTrue(
                    any(field in str(e['loc']) for e in errors for field in expected_fields),
                    f"Expected error for one of {expected_fields} not found in {errors}"
                )

    def test_decay_function_validation(self):
        """Test decay_function field validation."""
        # Valid decay functions
        valid_functions = ["exponential", "linear", "gaussian", "binary"]

        for func in valid_functions:
            with self.subTest(func):
                params = RecencyParameters(
                    recency_field="created_at",
                    decay_function=func
                )
                self.assertEqual(params.decay_function, func)

        # Invalid decay functions
        invalid_functions = [
            ("invalid", "invalid"),
            ("case_sensitive", "EXPONENTIAL"),
        ]

        for test_name, func in invalid_functions:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError):
                    RecencyParameters(
                        recency_field="created_at",
                        decay_function=func
                    )

    def test_scale_validation(self):
        """Test scale field validation."""
        # Valid scales
        valid_scales = [
            ("days", "7d", "7d"),
            ("hours", "24h", "24h"),
            ("decimal_days", "1.5d", "1.5d"),
            ("decimal_hours", "0.5h", "0.5h"),
        ]

        for test_name, scale, expected in valid_scales:
            with self.subTest(test_name):
                params = RecencyParameters(recency_field="created_at", scale=scale)
                self.assertEqual(params.scale, expected)

        # Invalid scales
        invalid_scales = [
            ("zero", "0d", "must be greater than 0"),
            ("negative", "-1d", "scale"),
            ("invalid_format", "7", "scale"),
            ("invalid_unit", "7w", "scale"),
            ("missing_number", "d", "scale"),
            ("whitespace", "7 d", "scale"),
        ]

        for test_name, scale, expected_error_msg in invalid_scales:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(recency_field="created_at", scale=scale)
                errors = exc_info.exception.errors()
                error_messages = [str(e['msg']) + str(e.get('loc', '')) for e in errors]
                self.assertTrue(
                    any(expected_error_msg in msg for msg in error_messages),
                    f"Expected '{expected_error_msg}' in error messages: {error_messages}"
                )

    def test_offset_validation(self):
        """Test offset field validation."""
        # Valid offsets
        valid_offsets = [
            ("zero_days", "0d", "0d"),
            ("days", "1d", "1d"),
            ("hours", "12h", "12h"),
            ("decimal_days", "0.5d", "0.5d"),
            ("decimal_hours", "1.5h", "1.5h"),
        ]

        for test_name, offset, expected in valid_offsets:
            with self.subTest(test_name):
                params = RecencyParameters(recency_field="created_at", offset=offset)
                self.assertEqual(params.offset, expected)

        # Invalid offsets
        invalid_offsets = [
            ("negative", "-1d"),
            ("invalid_format", "1"),
            ("invalid_unit", "1m"),  # minutes not supported
        ]

        for test_name, offset in invalid_offsets:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(recency_field="created_at", offset=offset)
                errors = exc_info.exception.errors()
                self.assertTrue(
                    any('offset' in str(e['loc']) for e in errors),
                    f"Expected 'offset' in error locations: {errors}"
                )

    def test_decay_to_validation(self):
        """Test decay_to field validation."""
        # Valid decay_to values
        valid_values = [
            ("default", 0.5, 0.5),
            ("min", 0.01, 0.01),
            ("max", 1.0, 1.0),
            ("near_max", 0.99, 0.99),
        ]

        for test_name, decay_to, expected in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(recency_field="created_at", decay_to=decay_to)
                self.assertEqual(params.decay_to, expected)

        # Invalid decay_to values
        invalid_values = [
            ("zero", 0.0),
            ("negative", -0.5),
            ("greater_than_one", 1.5),
        ]

        for test_name, decay_to in invalid_values:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(recency_field="created_at", decay_to=decay_to)
                errors = exc_info.exception.errors()
                self.assertTrue(
                    any('decay_to' in str(e['loc']) or 'decayTo' in str(e['loc']) for e in errors),
                    f"Expected 'decay_to' or 'decayTo' in error locations: {errors}"
                )

    def test_apply_in_ranking_phase_validation(self):
        """Test apply_in_ranking_phase field validation."""
        # Valid values
        valid_values = [
            ("all", "all"),
            ("only_global", "only-global"),
            ("exclude_global", "exclude-global"),
        ]

        for test_name, value in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    apply_in_ranking_phase=value
                )
                self.assertEqual(params.apply_in_ranking_phase, value)

        # Test default
        params = RecencyParameters(recency_field="created_at")
        self.assertEqual(params.apply_in_ranking_phase, "all")

        # Invalid value
        with self.assertRaises(ValidationError):
            RecencyParameters(
                recency_field="created_at",
                apply_in_ranking_phase="invalid"
            )

    def test_field_aliases(self):
        """Test field aliases (camelCase ↔ snake_case)."""
        alias_cases = [
            ("recency_field", {"recencyField": "created_at"}, "recency_field", "created_at"),
            ("decay_function", {"recency_field": "created_at", "decayFunction": "linear"}, "decay_function", "linear"),
            ("decay_to", {"recency_field": "created_at", "decayTo": 0.3}, "decay_to", 0.3),
            ("apply_in_ranking_phase", {"recency_field": "created_at", "applyInRankingPhase": "only-global"}, "apply_in_ranking_phase", "only-global"),
        ]

        for test_name, input_params, field_name, expected_value in alias_cases:
            with self.subTest(test_name):
                params = RecencyParameters(**input_params)
                self.assertEqual(getattr(params, field_name), expected_value)

    def test_default_values(self):
        """Test default values for all fields."""
        defaults = [
            ("decay_function", "exponential"),
            ("scale", "7d"),
            ("offset", "0d"),
            ("decay_to", 0.5),
            ("apply_in_ranking_phase", "all"),
        ]

        params = RecencyParameters(recency_field="created_at")

        for field_name, expected_value in defaults:
            with self.subTest(field_name):
                self.assertEqual(getattr(params, field_name), expected_value)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with self.assertRaises(ValidationError) as exc_info:
            RecencyParameters(
                recency_field="created_at",
                extraField="invalid"
            )

        errors = exc_info.exception.errors()
        self.assertTrue(any('extraField' in str(e['loc']) for e in errors))

    def test_serialization(self):
        """Test serialization and deserialization."""
        test_cases = [
            ("dict_serialization", {
                "recency_field": "created_at",
                "decay_function": "linear",
                "scale": "14d",
                "offset": "1d",
                "decay_to": 0.3,
                "apply_in_ranking_phase": "only-global"
            }),
            ("parse_from_dict", {
                "recency_field": "updated_at",
                "decay_function": "linear",
                "scale": "14d",
                "offset": "1d",
                "decay_to": 0.3,
                "apply_in_ranking_phase": "only-global"
            }),
            ("parse_with_aliases", {
                "recencyField": "updated_at",
                "decayFunction": "gaussian",
                "scale": "21d",
                "offset": "2d",
                "decayTo": 0.2,
                "applyInRankingPhase": "exclude-global"
            }),
        ]

        for test_name, data in test_cases:
            with self.subTest(test_name):
                if test_name == "dict_serialization":
                    params = RecencyParameters(**data)
                    result = params.dict()
                    for key, value in data.items():
                        self.assertEqual(result[key], value)
                elif test_name.startswith("parse"):
                    params = RecencyParameters.parse_obj(data)
                    # Verify fields using snake_case names
                    expected_values = {
                        "recency_field": data.get("recency_field") or data.get("recencyField"),
                        "decay_function": data.get("decay_function") or data.get("decayFunction"),
                        "scale": data.get("scale"),
                        "offset": data.get("offset"),
                        "decay_to": data.get("decay_to") or data.get("decayTo"),
                        "apply_in_ranking_phase": data.get("apply_in_ranking_phase") or data.get("applyInRankingPhase"),
                    }
                    for field, expected in expected_values.items():
                        self.assertEqual(getattr(params, field), expected)

        # Test dict with aliases
        with self.subTest("dict_with_alias"):
            params = RecencyParameters(
                recency_field="created_at",
                decay_function="exponential"
            )
            result = params.dict(by_alias=True)

            # Should use camelCase
            self.assertIn('recencyField', result)
            self.assertIn('decayFunction', result)
            self.assertIn('decayTo', result)
            self.assertIn('applyInRankingPhase', result)

        # Test JSON serialization
        with self.subTest("json_serialization"):
            params = RecencyParameters(
                recency_field="created_at",
                decay_function="gaussian",
                scale="7d",
                offset="0d",
                decay_to=0.5
            )
            json_str = params.json()

            self.assertIsInstance(json_str, str)
            self.assertIn("created_at", json_str)
            self.assertIn("gaussian", json_str)

    def test_add_to_score_weight_validation(self):
        """Test add_to_score_weight field validation."""
        # Valid add_to_score_weight values
        valid_values = [
            ("positive_small", 0.1, 0.1),
            ("positive_large", 10.0, 10.0),
            ("positive_one", 1.0, 1.0),
            ("positive_decimal", 0.01, 0.01),
        ]

        for test_name, add_to_score_weight, expected in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    add_to_score_weight=add_to_score_weight
                )
                self.assertEqual(params.add_to_score_weight, expected)

        # Invalid add_to_score_weight values
        invalid_values = [
            ("zero", 0.0),
            ("negative", -0.5),
            ("negative_large", -10.0),
        ]

        for test_name, add_to_score_weight in invalid_values:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(
                        recency_field="created_at",
                        add_to_score_weight=add_to_score_weight
                    )
                errors = exc_info.exception.errors()
                self.assertTrue(
                    any('add_to_score_weight' in str(e['loc']) or 'addToScoreWeight' in str(e['loc']) for e in errors),
                    f"Expected 'add_to_score_weight' or 'addToScoreWeight' in error locations: {errors}"
                )

    def test_add_to_score_weight_default_none(self):
        """Test add_to_score_weight defaults to None."""
        params = RecencyParameters(recency_field="created_at")
        self.assertIsNone(params.add_to_score_weight)

    def test_add_to_score_weight_alias(self):
        """Test add_to_score_weight alias (addToScoreWeight)."""
        params = RecencyParameters(
            recency_field="created_at",
            addToScoreWeight=0.5
        )
        self.assertEqual(params.add_to_score_weight, 0.5)

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('addToScoreWeight', result)
        self.assertEqual(result['addToScoreWeight'], 0.5)

    # ============= Grow Parameters Tests =============
    # Note: Grow parameters follow "all or nothing" rule - either all 4 must be
    # provided (growFrom, growFunction, growScale, growOffset) or none.

    def test_grow_params_all_or_nothing_validation(self):
        """Test that grow parameters must be either all provided or all omitted."""
        # Test partial combinations - all should fail
        partial_combinations = [
            # Only one param
            ("only_grow_from", {"grow_from": 0.5}),
            ("only_grow_function", {"grow_function": "exponential"}),
            ("only_grow_scale", {"grow_scale": "7d"}),
            ("only_grow_offset", {"grow_offset": "1d"}),
            # Two params
            ("grow_from_and_function", {"grow_from": 0.5, "grow_function": "exponential"}),
            ("grow_from_and_scale", {"grow_from": 0.5, "grow_scale": "7d"}),
            ("grow_from_and_offset", {"grow_from": 0.5, "grow_offset": "1d"}),
            ("grow_function_and_scale", {"grow_function": "exponential", "grow_scale": "7d"}),
            # Three params
            ("missing_grow_from", {"grow_function": "exponential", "grow_scale": "7d", "grow_offset": "1d"}),
            ("missing_grow_function", {"grow_from": 0.5, "grow_scale": "7d", "grow_offset": "1d"}),
            ("missing_grow_scale", {"grow_from": 0.5, "grow_function": "exponential", "grow_offset": "1d"}),
            ("missing_grow_offset", {"grow_from": 0.5, "grow_function": "exponential", "grow_scale": "7d"}),
        ]

        for test_name, grow_params in partial_combinations:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(recency_field="created_at", **grow_params)
                error_str = str(exc_info.exception)
                self.assertIn("all provided or all omitted", error_str.lower(),
                    f"Expected 'all provided or all omitted' in error: {error_str}")

        # All params provided - should pass
        with self.subTest("all_provided"):
            params = RecencyParameters(
                recency_field="created_at",
                grow_from=0.5,
                grow_function="exponential",
                grow_scale="7d",
                grow_offset="1d"
            )
            self.assertEqual(params.grow_from, 0.5)
            self.assertEqual(params.grow_function, "exponential")
            self.assertEqual(params.grow_scale, "7d")
            self.assertEqual(params.grow_offset, "1d")

        # No grow params - should pass
        with self.subTest("none_provided"):
            params = RecencyParameters(recency_field="created_at")
            self.assertIsNone(params.grow_from)
            self.assertIsNone(params.grow_function)
            self.assertIsNone(params.grow_scale)
            self.assertIsNone(params.grow_offset)

    def test_grow_from_validation(self):
        """Test grow_from field value validation."""
        # Valid grow_from values (must provide all grow params)
        valid_values = [
            ("min", 0.01),
            ("mid", 0.5),
            ("max", 1.0),
            ("near_min", 0.001),
        ]

        for test_name, grow_from in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=grow_from,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_from, grow_from)

        # Invalid grow_from values
        invalid_values = [
            ("zero", 0.0),
            ("negative", -0.5),
            ("greater_than_one", 1.5),
        ]

        for test_name, grow_from in invalid_values:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(
                        recency_field="created_at",
                        grow_from=grow_from,
                        grow_function="exponential",
                        grow_scale="7d",
                        grow_offset="0d"
                    )
                errors = exc_info.exception.errors()
                self.assertTrue(
                    any('grow_from' in str(e['loc']) or 'growFrom' in str(e['loc']) for e in errors),
                    f"Expected 'grow_from' or 'growFrom' in error locations: {errors}"
                )

    def test_grow_from_default_none(self):
        """Test grow_from defaults to None when no grow params provided."""
        params = RecencyParameters(recency_field="created_at")
        self.assertIsNone(params.grow_from)

    def test_grow_from_alias(self):
        """Test grow_from alias (growFrom)."""
        params = RecencyParameters(
            recency_field="created_at",
            growFrom=0.3,
            growFunction="exponential",
            growScale="7d",
            growOffset="0d"
        )
        self.assertEqual(params.grow_from, 0.3)

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('growFrom', result)
        self.assertEqual(result['growFrom'], 0.3)

    def test_grow_function_validation(self):
        """Test grow_function field validation."""
        # Valid grow functions (same as decay functions)
        valid_functions = ["exponential", "linear", "gaussian", "binary"]

        for func in valid_functions:
            with self.subTest(func):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=0.5,
                    grow_function=func,
                    grow_scale="7d",
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_function, func)

        # Invalid grow functions
        invalid_functions = [
            ("invalid", "invalid"),
            ("case_sensitive", "EXPONENTIAL"),
        ]

        for test_name, func in invalid_functions:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError):
                    RecencyParameters(
                        recency_field="created_at",
                        grow_from=0.5,
                        grow_function=func,
                        grow_scale="7d",
                        grow_offset="0d"
                    )

    def test_grow_function_alias(self):
        """Test grow_function alias (growFunction)."""
        params = RecencyParameters(
            recency_field="created_at",
            grow_from=0.5,
            growFunction="linear",
            grow_scale="7d",
            grow_offset="0d"
        )
        self.assertEqual(params.grow_function, "linear")

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('growFunction', result)
        self.assertEqual(result['growFunction'], 'linear')

    def test_grow_scale_validation(self):
        """Test grow_scale field validation."""
        # Valid scales (same format as scale)
        valid_scales = [
            ("days", "7d"),
            ("hours", "24h"),
            ("decimal_days", "1.5d"),
            ("decimal_hours", "0.5h"),
        ]

        for test_name, grow_scale in valid_scales:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=0.5,
                    grow_function="exponential",
                    grow_scale=grow_scale,
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_scale, grow_scale)

        # Invalid scales
        invalid_scales = [
            ("zero", "0d"),
            ("negative", "-1d"),
            ("invalid_format", "7"),
            ("invalid_unit", "7w"),
        ]

        for test_name, grow_scale in invalid_scales:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError):
                    RecencyParameters(
                        recency_field="created_at",
                        grow_from=0.5,
                        grow_function="exponential",
                        grow_scale=grow_scale,
                        grow_offset="0d"
                    )

    def test_grow_scale_alias(self):
        """Test grow_scale alias (growScale)."""
        params = RecencyParameters(
            recency_field="created_at",
            grow_from=0.5,
            grow_function="exponential",
            growScale="14d",
            grow_offset="0d"
        )
        self.assertEqual(params.grow_scale, "14d")

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('growScale', result)
        self.assertEqual(result['growScale'], '14d')

    def test_grow_offset_validation(self):
        """Test grow_offset field validation."""
        # Valid offsets (same format as offset)
        valid_offsets = [
            ("zero_days", "0d"),
            ("days", "1d"),
            ("hours", "12h"),
            ("decimal_days", "0.5d"),
        ]

        for test_name, grow_offset in valid_offsets:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    grow_from=0.5,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset=grow_offset
                )
                self.assertEqual(params.grow_offset, grow_offset)

        # Invalid offsets
        invalid_offsets = [
            ("negative", "-1d"),
            ("invalid_format", "1"),
            ("invalid_unit", "1m"),
        ]

        for test_name, grow_offset in invalid_offsets:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError):
                    RecencyParameters(
                        recency_field="created_at",
                        grow_from=0.5,
                        grow_function="exponential",
                        grow_scale="7d",
                        grow_offset=grow_offset
                    )

    def test_grow_offset_alias(self):
        """Test grow_offset alias (growOffset)."""
        params = RecencyParameters(
            recency_field="created_at",
            grow_from=0.5,
            grow_function="exponential",
            grow_scale="7d",
            growOffset="2d"
        )
        self.assertEqual(params.grow_offset, "2d")

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('growOffset', result)
        self.assertEqual(result['growOffset'], '2d')

    def test_grow_parameters_complete(self):
        """Test all grow parameters together."""
        params = RecencyParameters(
            recency_field="created_at",
            decay_function="exponential",
            scale="7d",
            offset="0d",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="linear",
            grow_scale="14d",
            grow_offset="1d"
        )

        self.assertEqual(params.grow_from, 0.3)
        self.assertEqual(params.grow_function, "linear")
        self.assertEqual(params.grow_scale, "14d")
        self.assertEqual(params.grow_offset, "1d")

    def test_grow_parameters_with_camel_case_aliases(self):
        """Test grow parameters using camelCase aliases."""
        params = RecencyParameters(
            recencyField="created_at",
            growFrom=0.4,
            growFunction="gaussian",
            growScale="21d",
            growOffset="3d"
        )

        self.assertEqual(params.recency_field, "created_at")
        self.assertEqual(params.grow_from, 0.4)
        self.assertEqual(params.grow_function, "gaussian")
        self.assertEqual(params.grow_scale, "21d")
        self.assertEqual(params.grow_offset, "3d")

    # ============= Center Parameter Tests =============

    def test_center_validation(self):
        """Test center field validation."""
        # Valid center values
        valid_values = [
            ("positive_float", 1709232000.0),
            ("zero", 0),
            ("large_value", 9999999999.9),
        ]

        for test_name, center in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    center=center
                )
                self.assertEqual(params.center, center)

        # Invalid center values
        invalid_values = [
            ("negative", -1.0),
            ("negative_large", -1709232000.0),
        ]

        for test_name, center in invalid_values:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(
                        recency_field="created_at",
                        center=center
                    )
                error_str = str(exc_info.exception)
                self.assertIn("center", error_str.lower())

    def test_center_default_none(self):
        """Test center defaults to None."""
        params = RecencyParameters(recency_field="created_at")
        self.assertIsNone(params.center)

    def test_center_alias(self):
        """Test center alias works."""
        params = RecencyParameters(
            recency_field="created_at",
            center=1709232000.0
        )
        self.assertEqual(params.center, 1709232000.0)

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('center', result)
        self.assertEqual(result['center'], 1709232000.0)

    # ============= ApplyToSubqueries Parameter Tests =============

    def test_apply_to_subqueries_validation(self):
        """Test apply_to_subqueries field validation."""
        # Valid values
        valid_values = [
            ("tensor_only", ["tensor"]),
            ("lexical_only", ["lexical"]),
            ("both", ["tensor", "lexical"]),
            ("empty_list", []),
        ]

        for test_name, apply_to in valid_values:
            with self.subTest(test_name):
                params = RecencyParameters(
                    recency_field="created_at",
                    apply_to_subqueries=apply_to
                )
                self.assertEqual(params.apply_to_subqueries, apply_to)

        # Invalid values
        invalid_values = [
            ("invalid_string", ["invalid"]),
            ("mixed_invalid", ["tensor", "invalid"]),
            ("uppercase", ["TENSOR"]),
        ]

        for test_name, apply_to in invalid_values:
            with self.subTest(test_name):
                with self.assertRaises(ValidationError) as exc_info:
                    RecencyParameters(
                        recency_field="created_at",
                        apply_to_subqueries=apply_to
                    )
                error_str = str(exc_info.exception)
                self.assertIn("apply_to_subqueries", error_str.lower().replace(" ", "_")
                              .replace("applytosubqueries", "apply_to_subqueries"))

    def test_apply_to_subqueries_default_none(self):
        """Test apply_to_subqueries defaults to None."""
        params = RecencyParameters(recency_field="created_at")
        self.assertIsNone(params.apply_to_subqueries)

    def test_apply_to_subqueries_alias(self):
        """Test applyToSubqueries alias works."""
        params = RecencyParameters(
            recency_field="created_at",
            applyToSubqueries=["tensor"]
        )
        self.assertEqual(params.apply_to_subqueries, ["tensor"])

        # Test dict serialization with alias
        result = params.dict(by_alias=True)
        self.assertIn('applyToSubqueries', result)
        self.assertEqual(result['applyToSubqueries'], ["tensor"])
