from enum import Enum
from typing import List, Literal, Optional

from numpy import dtype
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..base_model_properties import BaseModelProperties, TritonModelProperties
from ..data_type_conversion import convert_to_numpy_dtype, convert_to_triton_data_type


class ImagePreprocessor(str, Enum):
    SigLIP = "SigLIP"
    OpenAI = "OpenAI"
    OpenCLIP = "OpenCLIP"
    # MobileCLIP = "MobileCLIP" # TODO Add this back when we upgrade the open clip and torch versions
    CLIPA = "CLIPA"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class OpenCLIPTritonModelProperties(TritonModelProperties):
    @field_validator("input", mode="after")
    @classmethod
    def validate_input(cls, v: list):
        input_length = len(v)
        if input_length != 1:
            raise ValueError(
                f"For OpenCLIP models, triton_text_encoder must have exactly 1 input. "
                f"Received '{v}' with '{input_length}' inputs."
            )

        if v[0].name != "input":
            raise ValueError(
                f"For OpenCLIP models, triton_text_encoder input must be named 'input'. "
                f"Received '{v[0].name}' instead."
            )
        return v

    @field_validator("output", mode="after")
    @classmethod
    def validate_output(cls, v: list):
        output_length = len(v)
        if output_length != 1:
            raise ValueError(
                f"For OpenCLIP models, triton_text_encoder must have exactly 1 output. "
                f"Received '{v}' with '{output_length}' inputs."
            )

        if v[0].name != "output":
            raise ValueError(
                f"For OpenCLIP models, triton_text_encoder output must be named 'output'. "
                f"Received '{v[0].name}' instead."
            )
        return v


class OpenCLIPModelProperties(BaseModelProperties):
    """
    A class to represent the properties of an OpenCLIP model.

    Attributes:
        type: The type of the model. It should be 'open_clip'.
        tokenizer: The name of the tokenizer. It is optional.
        image_preprocessor: The image preprocessor used by the model. It should be one of the values in the
            ImagePreprocessor enum.
        mean: The mean values for the image preprocessor. It is optional. It provided, it will override the
            default mean values of the image preprocessor.
        std: The standard deviation values for the image preprocessor. It is optional. It provided, it will
            override the default standard deviation values of the image preprocessor.
        size: The size of the image. It is optional. If provided, it will override the default size of the image.
        note: A note about the model. It is optional.
    """

    tokenizer: Optional[str] = None
    image_preprocessor: ImagePreprocessor = Field(
        default=ImagePreprocessor.OpenCLIP, alias="imagePreprocessor"
    )
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: Optional[int] = None
    note: Optional[str] = None
    type: Literal["open_clip"]
    triton_text_encoder_properties: TritonModelProperties = Field(
        ..., alias="tritonTextEncoderProperties"
    )
    triton_image_encoder_properties: TritonModelProperties = Field(
        ..., alias="tritonImageEncoderProperties"
    )

    _text_input_numpy_type: dtype = PrivateAttr()
    _text_input_triton_type: str = PrivateAttr()
    _image_input_numpy_type: dtype = PrivateAttr()
    _image_input_triton_type: str = PrivateAttr()

    @model_validator(mode="after")
    def _cache_derived_types(self):
        """
        Cache the derived types for the model properties. This cache will be the hot path for encoding so
        we use a model validator to do this once at initialization time.
        """
        self._text_input_numpy_type = convert_to_numpy_dtype(
            self.triton_text_encoder_properties.input[0].data_type
        )
        self._text_input_triton_type = convert_to_triton_data_type(
            self.triton_text_encoder_properties.input[0].data_type
        )
        self._image_input_numpy_type = convert_to_numpy_dtype(
            self.triton_image_encoder_properties.input[0].data_type
        )
        self._image_input_triton_type = convert_to_triton_data_type(
            self.triton_image_encoder_properties.input[0].data_type
        )
        return self

    @property
    def text_input_numpy_type(self) -> dtype:
        return self._text_input_numpy_type

    @property
    def text_input_triton_type(self) -> str:
        return self._text_input_triton_type

    @property
    def image_input_numpy_type(self) -> dtype:
        return self._image_input_numpy_type

    @property
    def image_input_triton_type(self) -> str:
        return self._image_input_triton_type
