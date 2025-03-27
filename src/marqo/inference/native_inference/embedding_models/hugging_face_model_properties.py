import json
from enum import Enum
from json import JSONDecodeError
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from pydantic import Field, validator, root_validator

from marqo.base_model import ImmutableBaseModel
from marqo.inference.native_inference.embedding_models.marqo_base_model_properties import MarqoBaseModelProperties
from marqo.s2_inference.configs import ModelCache
from marqo.logging import get_logger
from marqo.tensor_search.models.private_models import ModelLocation

logger = get_logger(__name__)


class PoolingMethod(str, Enum):
    Mean = "mean"
    CLS = "cls"


class HuggingFaceModelFlags(ImmutableBaseModel):
    """
    Flags passed to transformers.AutoModel.from_pretrained()
    """
    trust_remote_code: Optional[bool] = None
    use_memory_efficient_attention: Optional[bool] = None
    unpad_inputs: Optional[bool] = None


class HuggingFaceTokenizerFlags(ImmutableBaseModel):
    """
    Flags passed to transformers.AutoTokenizer.from_pretrained()
    """
    trust_remote_code: Optional[bool] = None


class HuggingFaceModelProperties(MarqoBaseModelProperties):
    """
    A class to represent the properties of a Hugging Face model.

    Attributes:
        name: The name of the model. This will be used as the repo_id in the Hugging Face model hub.
            This attribute is neglected if 'url' or 'model_location' is provided.
            We are not raising an error right now as that would be a breaking change.
        tokens: The token length of the model. It is default to 128.
        type: The type of the model. It should be "hf".
        url: The URL of the model checkpoint. It is optional.
        dimensions: The dimensions of the model.
        model_location: The location of the model. It is optional.
        note: A note about the model. It is optional.
        pooling_method: The pooling method for the model. It should be one of the values in the PoolingMethod enum.
        trust_remote_code: Allow remote code execution.
    """
    name: Optional[str] = None
    tokens: int = 128
    url: Optional[str] = None
    model_location: Optional[ModelLocation] = Field(default=None, alias="modelLocation")
    note: Optional[str] = None
    pooling_method: PoolingMethod = Field(..., alias="poolingMethod")
    trust_remote_code: bool = Field(False, alias="trustRemoteCode")

    @validator("type")
    def _validate_type(cls, v):
        if v not in ["hf", "hf_stella"]:
            raise ValueError("The type of the model should be 'hf' or 'hf_stella'.")
        return v

    @root_validator(pre=True, skip_on_failure=True)
    def _validate_or_infer_pooling_method(cls, values):
        """Infer the pooling_method from the model name if it is not provided.

        If the pooling_method is provided, return the values as is.
        """
        pooling_method = values.get("pooling_method") or values.get("poolingMethod")
        if pooling_method is not None:
            return values
        name = values.get('name')
        if isinstance(name, str) and name:
            pooling_method = cls._infer_pooling_method_from_name(name)
        else:
            pooling_method = PoolingMethod.Mean
        values["pooling_method"] = pooling_method
        return values

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

        def log_warning_and_return_default():
            logger.warning(f"Could not infer pooling method from the model {name}. Defaulting to mean pooling.")
            return PoolingMethod.Mean

        try:
            file_path = hf_hub_download(repo_id, file_name, cache_dir=ModelCache.hf_cache_path)
        except HfHubHTTPError:
            return log_warning_and_return_default()

        try:
            with open(file_path, 'r') as file:
                content = json.loads(file.read())
        except JSONDecodeError:
            return log_warning_and_return_default()

        if not isinstance(content, dict):
            return log_warning_and_return_default()

        if content.get("pooling_mode_cls_token") is True:
            return PoolingMethod.CLS
        elif content.get("pooling_mode_mean_tokens") is True:
            return PoolingMethod.Mean
        else:
            return log_warning_and_return_default()

    @root_validator(skip_on_failure=True)
    def _validate_minimum_required_fields_to_load(cls, values):
        """
        Validate that at least one of 'name', 'url', or 'model_location' is provided.
        But 'url' and 'model_location' should not be provided together.
        """
        if values.get("url") and values.get("model_location"):
            raise ValueError("Only one of 'url' and 'model_location' should be provided.")
        is_custom = values.get("url") or values.get("model_location")
        if not values.get("name") and not is_custom:
            raise ValueError("At least one of 'name', 'url', or 'model_location' should be provided.")
        return values
