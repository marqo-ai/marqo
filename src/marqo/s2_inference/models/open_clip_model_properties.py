
from enum import Enum
from typing import Optional, List

from pydantic import Field

from marqo.base_model import MarqoBaseModel
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth


class ImagePreprocessor(str, Enum):
    SiCLIP = "SiCLIP"
    OpenAI = "OpenAI"
    OpenCLIP = "OpenCLIP"
    MobileCLIP = "MobileCLIP"
    CLIPA = "CLIPA"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class OpenCLIPModelProperties(MarqoBaseModel):
    name: str
    type: str
    jit: bool = False
    precision: Precision = Precision.FP32
    url: Optional[str] = None
    model_location: Optional[ModelLocation] = Field(default=None, alias="modelLocation")
    tokenizer: Optional[str] = None
    model_auth: Optional[ModelAuth] = Field(default=None, alias="modelAuth")
    image_preprocessor: ImagePreprocessor = Field(default=ImagePreprocessor.OpenCLIP, alias="imagePreprocessor")
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: Optional[int] = None