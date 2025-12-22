import numpy as np

from .base_model_properties import DataType


def convert_to_triton_data_type(dtype: DataType) -> str:
    """
    Convert DataType enum to corresponding Triton data type string.

    Args:
        dtype (DataType): The data type to convert.

    Returns:
        str: The corresponding Triton data type string.

    Raises:
        ValueError: If the provided data type is not supported.
    """
    mapping = {
        DataType.TYPE_FP64: "FP64",
        DataType.TYPE_FP32: "FP32",
        DataType.TYPE_FP16: "FP16",
        DataType.TYPE_INT8: "INT8",
        DataType.TYPE_INT16: "INT16",
        DataType.TYPE_INT32: "INT32",
        DataType.TYPE_INT64: "INT64",
        DataType.TYPE_BF16: "BF16",
    }
    if dtype in mapping:
        return mapping[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def convert_to_numpy_dtype(dtype: DataType):
    """
    Convert DataType enum to corresponding numpy data type.

    Args:
        dtype (DataType): The data type to convert.

    Returns:
        numpy.dtype: The corresponding numpy data type.

    Raises:
        ValueError: If the provided data type is not supported.
    """

    mapping = {
        DataType.TYPE_FP64: np.float64,
        DataType.TYPE_FP32: np.float32,
        DataType.TYPE_FP16: np.float16,
        DataType.TYPE_INT8: np.int8,
        DataType.TYPE_INT16: np.int16,
        DataType.TYPE_INT32: np.int32,
        DataType.TYPE_INT64: np.int64,
        DataType.TYPE_BF16: np.uint16,  # Note: NumPy does not have a native bfloat16 type
    }
    if dtype in mapping:
        return mapping[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}")
