import json
from enum import Enum
from json import JSONDecodeError
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from pydantic import Field, validator, root_validator

from marqo.base_model import MarqoBaseModel
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.logger import get_logger
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth

logger = get_logger(__name__)


class PoolingMethod(str, Enum):
    Mean = "mean"
    CLS = "cls"


class HuggingFaceModelProperties(MarqoBaseModel):
    """
    A class to represent the properties of a Hugging Face model.

    Attributes:
        name: The name of the model. This will be used as the repo_id in the Hugging Face model hub.
        token: The token length of the model. It is default to 128.
        type: The type of the model. It should be "hf".
        url: The URL of the model checkpoint. It is optional.
        model_location: The location of the model. It is optional.
        model_auth: The authentication information for the model. It is optional.
        note: A note about the model. It is optional.
        pooling_method: The pooling method for the model. It should be one of the values in the PoolingMethod enum.
    """
    name: Optional[str] = None
    token: int = 128
    type: str
    url: Optional[str] = None
    model_location: Optional[ModelLocation] = Field(default=None, alias="modelLocation")
    model_auth: Optional[ModelAuth] = Field(default=None, alias="modelAuth")
    note: Optional[str] = None
    pooling_method: PoolingMethod = Field(default=PoolingMethod.Mean, alias="poolingMethod")

    @validator("type")
    def _validate_type(cls, v):
        if v != "hf":
            raise ValueError("The type of the model should be 'hf'.")
        return v

    @validator('pooling_method', pre=True, always=True)
    def validate_or_infer_pooling_method(cls, v, values):
        if v is not None:
            return v
        name = values.get('name')
        if name and isinstance(name, str):
            return cls._infer_pooling_method_from_name(name)
        return PoolingMethod.Mean

    @staticmethod
    def _infer_pooling_method_from_name(name: str) -> PoolingMethod:
        """
        Infer the pooling method from the model name.
        Args:
            name: The name of the model. This is the repo_id in the Hugging Face model hub.

        Returns:
            The inferred pooling method.
        """
        repo_id = name
        file_name = "1_Pooling/config.json"
        try:
            file_path = hf_hub_download(repo_id, file_name, cache_dir=ModelCache.hf_cache_path)
        except HfHubHTTPError:
            logger.warn(f"Could not infer pooling method from the model {name}. Defaulting to mean pooling.")
            return PoolingMethod.Mean

        try:
            with open(file_path, 'r') as file:
                content = json.loads(file.read())
        except JSONDecodeError:
            logger.warn(f"Could not infer pooling method from the model {name}. Defaulting to mean pooling.")
            return PoolingMethod.Mean

        if not isinstance(content, dict):
            logger.warn(f"Could not infer pooling method from the model {name}. Defaulting to mean pooling.")
            return PoolingMethod.Mean

        if content.get("pooling_mode_cls_token") is True:
            return PoolingMethod.CLS
        elif content.get("pooling_mode_mean_tokens") is True:
            return PoolingMethod.Mean
        else:
            logger.warn(f"Could not infer pooling method from the model {name}. Defaulting to mean pooling.")
            return PoolingMethod.Mean


    @root_validator(pre=True)
    def _validate_url_and_model_location(cls, values):
        if values.get("url") and values.get("model_location"):
            raise ValueError("Only one of 'url' and 'model_location' should be provided.")
        return values