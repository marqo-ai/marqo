from enum import Enum
from pydantic import Field
from typing import Optional, List, Literal

from ..base_model_properties import BaseModelProperties, TritonModelProperties


class ImagePreprocessor(str, Enum):
    SigLIP = "SigLIP"
    OpenAI = "OpenAI"
    OpenCLIP = "OpenCLIP"
    # MobileCLIP = "MobileCLIP" # TODO Add this back when we upgrade the open clip and torch versions
    CLIPA = "CLIPA"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class OpenCLIPModelProperties(BaseModelProperties):
    """
    A class to represent the properties of an OpenCLIP model.

    Attributes:
        name: The name of the model. It will be used to load the image preprocessor/tokenizer.
        type: The type of the model. It should be 'open_clip'.
        jit: A boolean indicating whether the model is JIT compiled.
        precision: The precision of the model. It should be either 'fp32' or 'fp16'.
        url: The URL of the model checkpoint. It is optional.
        localpath: The local path of the model checkpoint. It is optional.
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
    name: str
    jit: bool = False
    precision: Precision = Precision.FP32
    url: Optional[str] = None
    localpath: Optional[str] = None
    tokenizer: Optional[str] = None
    image_preprocessor: ImagePreprocessor = Field(default=ImagePreprocessor.OpenCLIP, alias="imagePreprocessor")
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: Optional[int] = None
    note: Optional[str] = None
    pretrained: Optional[str] = None
    type: Literal["open_clip"]
    tritonTextEncoder: TritonModelProperties
    tritonImageEncoder: TritonModelProperties
