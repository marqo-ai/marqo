
from enum import Enum
from typing import Optional, List

from pydantic import Field

from marqo.base_model import MarqoBaseModel
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth


class ImagePreprocessor(str, Enum):
    SigLIP = "SigLIP"
    OpenAI = "OpenAI"
    OpenCLIP = "OpenCLIP"
    MobileCLIP = "MobileCLIP"
    CLIPA = "CLIPA"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class OpenCLIPModelProperties(MarqoBaseModel):
    """
    A class to represent the properties of an OpenCLIP model.

    Attributes:
        name: The name of the model. It can be the name of the model for loading information. e.g., the
            architecture name of the model, the name of the model in the Hugging Face model hub, etc. It might be
            the same as the model tag but this is not necessary.
        type: The type of the model. It should be 'open_clip'.
        jit: A boolean indicating whether the model is JIT compiled.
        precision: The precision of the model. It should be either 'fp32' or 'fp16'.
        url: The URL of the model checkpoint. It is optional.
        model_location: The location of the model. It is optional.
        tokenizer: The name of the tokenizer. It is optional.
        model_auth: The authentication information for the model. It is optional.
        image_preprocessor: The image preprocessor used by the model. It should be one of the values in the
            ImagePreprocessor enum.
        mean: The mean values for the image preprocessor. It is optional. It provided, it will override the
            default mean values of the image preprocessor.
        std: The standard deviation values for the image preprocessor. It is optional. It provided, it will
            override the default standard deviation values of the image preprocessor.
        size: The size of the image. It is optional. If provided, it will override the default size of the image.
        note: A note about the model. It is optional.
        pretrained: The name of the pretrained model. It is optional.
    """
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
    note: Optional[str] = None
    pretrained: Optional[str] = None