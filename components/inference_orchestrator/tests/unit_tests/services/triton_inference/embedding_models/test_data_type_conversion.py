import unittest

import numpy as np

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    DataType,
)
from inference_orchestrator.services.triton_inference.embedding_models.data_type_conversion import (
    convert_to_numpy_dtype,
    convert_to_triton_data_type,
)


class TestConvertToTritonDataType(unittest.TestCase):
    """Tests for convert_to_triton_data_type function."""

    def test_convert_to_triton_data_type_all_valid_types(self):
        """Test conversion of all valid DataType values to Triton types."""
        test_cases = [
            ("FP64", DataType.TYPE_FP64, "FP64"),
            ("FP32", DataType.TYPE_FP32, "FP32"),
            ("FP16", DataType.TYPE_FP16, "FP16"),
            ("INT8", DataType.TYPE_INT8, "INT8"),
            ("INT16", DataType.TYPE_INT16, "INT16"),
            ("INT32", DataType.TYPE_INT32, "INT32"),
            ("INT64", DataType.TYPE_INT64, "INT64"),
            ("BF16", DataType.TYPE_BF16, "BF16"),
        ]

        for msg, input_dtype, expected_output in test_cases:
            with self.subTest(msg=msg):
                result = convert_to_triton_data_type(input_dtype)
                self.assertEqual(expected_output, result)
                self.assertIsInstance(result, str)

    def test_convert_to_triton_data_type_invalid_type_raises_error(self):
        """Test that invalid DataType raises ValueError."""
        # Create a mock object that's not a valid DataType
        invalid_dtype = "INVALID_TYPE"
        with self.assertRaises(ValueError) as context:
            convert_to_triton_data_type(invalid_dtype)
        self.assertIn("Unsupported data type", str(context.exception))


class TestConvertToNumpyDtype(unittest.TestCase):
    """Tests for convert_to_numpy_dtype function."""

    def test_convert_to_numpy_dtype_all_valid_types(self):
        """Test conversion of all valid DataType values to numpy dtypes."""
        test_cases = [
            ("FP64", DataType.TYPE_FP64, np.float64),
            ("FP32", DataType.TYPE_FP32, np.float32),
            ("FP16", DataType.TYPE_FP16, np.float16),
            ("INT8", DataType.TYPE_INT8, np.int8),
            ("INT16", DataType.TYPE_INT16, np.int16),
            ("INT32", DataType.TYPE_INT32, np.int32),
            ("INT64", DataType.TYPE_INT64, np.int64),
            ("BF16", DataType.TYPE_BF16, np.uint16),
        ]

        for msg, input_dtype, expected_output in test_cases:
            with self.subTest(msg=msg):
                result = convert_to_numpy_dtype(input_dtype)
                self.assertEqual(expected_output, result)

    def test_convert_to_numpy_dtype_invalid_type_raises_error(self):
        """Test that invalid DataType raises ValueError."""
        invalid_dtype = "INVALID_TYPE"
        with self.assertRaises(ValueError) as context:
            convert_to_numpy_dtype(invalid_dtype)
        self.assertIn("Unsupported data type", str(context.exception))

    def test_convert_to_numpy_dtype_bfloat16_uses_uint16(self):
        """Test that BF16 maps to np.uint16 since NumPy has no native bfloat16."""
        result = convert_to_numpy_dtype(DataType.TYPE_BF16)
        self.assertEqual(np.uint16, result)
