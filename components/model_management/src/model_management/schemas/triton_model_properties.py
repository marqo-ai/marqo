from enum import StrEnum

from model_management.schemas.app_models import AppBaseModel
from pydantic import Field, field_validator

from ..services.model_manager.url_parser import get_base_filename


class DataType(StrEnum):
    TYPE_FP64 = "TYPE_FP64"
    TYPE_FP32 = "TYPE_FP32"
    TYPE_FP16 = "TYPE_FP16"
    TYPE_INT8 = "TYPE_INT8"
    TYPE_INT16 = "TYPE_INT16"
    TYPE_INT32 = "TYPE_INT32"
    TYPE_INT64 = "TYPE_INT64"
    TYPE_BF16 = "TYPE_BF16"


class ModelInput(AppBaseModel):
    """
    ModelInput defines the input of the model.

    Attributes:
        name (str): The name of the input.
        dims (list[int]): The dimensions of the input tensor.
        data_type (DataType): The data type of the input tensor.
    """

    name: str
    dims: list[int]
    data_type: DataType = Field(..., validation_alias="dataType")


class ModelOutput(AppBaseModel):
    """
    ModelOutput defines the output of the model.

    Attributes:
        name (str): The name of the output.
        dims (list[int]): The dimensions of the output tensor.
        data_type (DataType): The data type of the output tensor.
    """

    name: str
    dims: list[int]
    data_type: DataType = Field(..., validation_alias="dataType")


class TritonModelProperties(AppBaseModel):
    """
    TritonModelProperties defines the properties of a model to be served by Triton Inference Server.

    Attributes:
        name (str): The name of the model.
        max_batch_size (int): The maximum batch size for the model. Default is 8
        sources (list[str]): A list of sources for the model. It can be a local path, a URL, or a s3 URI.
            1 to 5 sources can be provided if the model consists of multiple files.
        input (list[ModelInput]): A list of input definitions for the model.
        output (list[ModelOutput]): A list of output definitions for the model. Currently only
            supports a single output for embeddings models.
    """

    name: str
    max_batch_size: int = Field(8, validation_alias="maxBatchSize", gt=0, le=128)
    sources: list[str] = Field(
        ..., validation_alias="sources", min_length=1, max_length=5
    )
    input: list[ModelInput] = Field(..., validation_alias="input")
    output: list[ModelOutput] = Field(
        ..., validation_alias="output", min_length=1, max_length=1
    )

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
