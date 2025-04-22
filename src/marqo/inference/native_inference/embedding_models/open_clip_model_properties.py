from enum import Enum
from typing import Optional, List

from pydantic import Field, root_validator

from marqo.inference.native_inference.embedding_models.marqo_base_model_properties import MarqoBaseModelProperties
from marqo.tensor_search.models.private_models import ModelLocation


class ImagePreprocessor(str, Enum):
    SigLIP = "SigLIP"
    OpenAI = "OpenAI"
    OpenCLIP = "OpenCLIP"
    # MobileCLIP = "MobileCLIP" # TODO Add this back when we upgrade the open clip and torch versions
    CLIPA = "CLIPA"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class OpenCLIPModelProperties(MarqoBaseModelProperties):
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
        localpath: The local path of the model checkpoint. It is optional.
        model_location: The location of the model. It is optional.
        tokenizer: The name of the tokenizer. It is optional.
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
    jit: bool = False
    precision: Precision = Precision.FP32
    url: Optional[str] = None
    localpath: Optional[str] = None
    model_location: Optional[ModelLocation] = Field(default=None, alias="modelLocation")
    tokenizer: Optional[str] = None
    image_preprocessor: ImagePreprocessor = Field(default=ImagePreprocessor.OpenCLIP, alias="imagePreprocessor")
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: Optional[int] = None
    note: Optional[str] = None
    pretrained: Optional[str] = None

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_custom_loading_fields(cls, values):
        url = values.get("url")
        localpath = values.get("localpath")
        model_location = values.get("model_location")

        provided_fields = sum(1 for field in [url, localpath, model_location] if field is not None)
        if provided_fields > 1:
            raise ValueError("Only one of 'url', 'localpath', or 'model_location' should be provided.")

        return values
