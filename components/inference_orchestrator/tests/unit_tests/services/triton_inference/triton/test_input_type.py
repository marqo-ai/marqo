import unittest

import numpy as np

from inference_orchestrator.services.triton_inference.triton.input_type import (
    InputType,
    InputTypeEnum,
)


class TestInputType(unittest.TestCase):
    """Tests for InputType dataclass."""

    def test_input_type_creation(self):
        """Test InputType can be created with code and dtype."""
        input_type = InputType("FP32", np.float32)
        self.assertEqual("FP32", input_type.code)
        self.assertEqual(np.float32, input_type.dtype)

    def test_input_type_frozen(self):
        """Test InputType is frozen (immutable)."""
        input_type = InputType("FP32", np.float32)
        with self.assertRaises(AttributeError):
            input_type.code = "FP16"


class TestInputTypeEnum(unittest.TestCase):
    """Tests for InputTypeEnum."""

    def test_input_type_enum_values(self):
        """Test all InputTypeEnum values have correct code and dtype."""
        test_cases = [
            ("INT32", InputTypeEnum.INT32, "INT32", np.int32),
            ("INT64", InputTypeEnum.INT64, "INT64", np.int64),
            ("FP32", InputTypeEnum.FP32, "FP32", np.float32),
            ("FP16", InputTypeEnum.FP16, "FP16", np.float16),
            ("UINT8", InputTypeEnum.UINT8, "UINT8", np.uint8),
        ]

        for msg, enum_value, expected_code, expected_dtype in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(expected_code, enum_value.value.code)
                self.assertEqual(expected_dtype, enum_value.value.dtype)

    def test_input_type_enum_access(self):
        """Test InputTypeEnum values can be accessed correctly."""
        self.assertIsInstance(InputTypeEnum.FP32, InputTypeEnum)
        self.assertIsInstance(InputTypeEnum.FP32.value, InputType)

    def test_input_type_enum_all_members(self):
        """Test InputTypeEnum has all expected members."""
        expected_members = {"INT32", "INT64", "FP32", "FP16", "UINT8"}
        actual_members = {member.name for member in InputTypeEnum}
        self.assertEqual(expected_members, actual_members)
