from enum import Enum
from typing import Literal, Optional

from pydantic import Field, field_validator

from inference_orchestrator.schemas.base_model import AppImmutableBaseModel

from .url_parser import get_base_filename


class DataType(str, Enum):
    TYPE_FP64 = "TYPE_FP64"
    TYPE_FP32 = "TYPE_FP32"
    TYPE_FP16 = "TYPE_FP16"
    TYPE_INT8 = "TYPE_INT8"
    TYPE_INT16 = "TYPE_INT16"
    TYPE_INT32 = "TYPE_INT32"
    TYPE_INT64 = "TYPE_INT64"
    TYPE_BF16 = "TYPE_BF16"


class BaseModelProperties(AppImmutableBaseModel):
    """
    The base class for all model properties classes.

    Attributes:
        name: The name of the model.
        triton_model_name: An optional override name used by the inference orchestrator for loading
            tokenizers/preprocessors. When set, effective_name returns this value instead of name.
            This supports safe migration where the Vespa-stored 'name' must remain unchanged for
            old pod cache key stability, while the new orchestrator needs a different name.
        dimensions: The dimensions of the model.
        type: The type of the model
    """

    name: str
    triton_model_name: Optional[str] = Field(default=None, alias="tritonModelName")
    dimensions: int = Field(..., ge=1)
    type: Literal["open_clip", "hf", "random"]

    @property
    def effective_name(self) -> str:
        """Return tritonModelName if set, otherwise fall back to name."""
        return self.triton_model_name if self.triton_model_name is not None else self.name


class ModelInput(AppImmutableBaseModel):
    """
    ModelInput defines the input of the model.

    Attributes:
        name (str): The name of the input.
        dims (list[int]): The dimensions of the input tensor.
        data_type (DataType): The data type of the input tensor.
    """

    name: str
    dims: list[int]
    data_type: DataType = Field(..., alias="dataType")


class ModelOutput(AppImmutableBaseModel):
    """
    ModelOutput defines the output of the model.

    Attributes:
        name (str): The name of the output.
        dims (list[int]): The dimensions of the output tensor.
        data_type (DataType): The data type of the output tensor.
    """

    name: str
    dims: list[int]
    data_type: DataType = Field(..., alias="dataType")


class TritonModelProperties(AppImmutableBaseModel):
    """
    TritonModelProperties defines the properties of a model to be served by Triton Inference Server.

    For backward compatibility, the limit on each field can only be relaxed, not tightened.

    Attributes:
        name (str): The name of the model.
        max_batch_size (int): The maximum batch size for the model. Default is 8
        sources (list[str]): A list of sources for the model. It can be a local path, a URL, or a s3 URI.
            1 to 5 sources can be provided if the model consists of multiple files.
        output (list[ModelOutput]): A list of output definitions for the model. Currently only
            supports a single output for embeddings models.
        input (list[ModelInput]): A list of input definitions for the model. Supports 1 to 3 inputs.
    """

    name: str
    max_batch_size: int = Field(8, alias="maxBatchSize", gt=0, le=128)
    sources: list[str] = Field(..., alias="sources", min_length=1, max_length=5)
    output: list[ModelOutput] = Field(..., alias="output", min_length=1, max_length=1)
    input: list[ModelInput] = Field(..., alias="input", min_length=1, max_length=3)

    @field_validator("sources", mode="after")
    @classmethod
    def _validate_sources(cls, values: list[str]) -> list[str]:
        """All sources must point to a model.onnx file, or a model.onnx.data file."""
        for v in values:
            base_filename = get_base_filename(v)
            if not (
                base_filename == "model.onnx"
                or base_filename.startswith("model.onnx.data")
            ):
                raise ValueError(
                    f"All sources must point to a model.onnx file, or a model.onnx.data file. "
                    f"Received invalid source: {v}"
                )
        return values
