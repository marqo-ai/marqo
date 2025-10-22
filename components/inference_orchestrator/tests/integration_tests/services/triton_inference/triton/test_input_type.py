"""
Integration tests for InputType and InputTypeEnum.

This module tests the InputType dataclass and InputTypeEnum to ensure they correctly
map Triton input type codes to numpy dtypes.
"""

import unittest

import numpy as np

from inference_orchestrator.services.triton_inference.triton.input_type import (
    InputType,
    InputTypeEnum,
)


class TestInputType(unittest.TestCase):
    """Test suite for InputType and InputTypeEnum."""

    def test_input_type_enum_values(self):
        """Test that all InputTypeEnum values are correctly defined."""
        test_cases = [
            ("INT32", InputTypeEnum.INT32, "INT32", np.int32),
            ("INT64", InputTypeEnum.INT64, "INT64", np.int64),
            ("FP32", InputTypeEnum.FP32, "FP32", np.float32),
            ("FP16", InputTypeEnum.FP16, "FP16", np.float16),
            ("UINT8", InputTypeEnum.UINT8, "UINT8", np.uint8),
        ]

        for enum_name, enum_value, expected_code, expected_dtype in test_cases:
            with self.subTest(enum_name=enum_name):
                self.assertIsInstance(enum_value.value, InputType)
                self.assertEqual(enum_value.value.code, expected_code)
                self.assertEqual(enum_value.value.dtype, expected_dtype)

    def test_input_type_frozen(self):
        """Test that InputType is immutable (frozen dataclass)."""
        input_type = InputTypeEnum.INT32.value

        with self.assertRaises(Exception):
            input_type.code = "MODIFIED"

    def test_input_type_dtypes_match_numpy(self):
        """Test that all dtype values are valid numpy types."""
        for enum_member in InputTypeEnum:
            with self.subTest(enum_name=enum_member.name):
                dtype = enum_member.value.dtype
                # Verify we can create a numpy array with this dtype
                arr = np.array([1, 2, 3], dtype=dtype)
                self.assertEqual(arr.dtype, dtype)

    def test_input_type_code_access(self):
        """Test accessing code attribute from InputType."""
        test_cases = [
            (InputTypeEnum.INT32, "INT32"),
            (InputTypeEnum.INT64, "INT64"),
            (InputTypeEnum.FP32, "FP32"),
            (InputTypeEnum.FP16, "FP16"),
            (InputTypeEnum.UINT8, "UINT8"),
        ]

        for enum_value, expected_code in test_cases:
            with self.subTest(expected_code=expected_code):
                self.assertEqual(enum_value.value.code, expected_code)

    def test_input_type_dtype_access(self):
        """Test accessing dtype attribute from InputType."""
        test_cases = [
            (InputTypeEnum.INT32, np.int32),
            (InputTypeEnum.INT64, np.int64),
            (InputTypeEnum.FP32, np.float32),
            (InputTypeEnum.FP16, np.float16),
            (InputTypeEnum.UINT8, np.uint8),
        ]

        for enum_value, expected_dtype in test_cases:
            with self.subTest(dtype=expected_dtype.__name__):
                self.assertEqual(enum_value.value.dtype, expected_dtype)

    def test_enum_member_count(self):
        """Test that InputTypeEnum has the expected number of members."""
        self.assertEqual(len(InputTypeEnum), 5)

    def test_enum_member_names(self):
        """Test that all expected enum member names exist."""
        expected_names = {"INT32", "INT64", "FP32", "FP16", "UINT8"}
        actual_names = {member.name for member in InputTypeEnum}
        self.assertEqual(actual_names, expected_names)


if __name__ == "__main__":
    unittest.main()
