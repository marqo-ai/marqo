from enum import Enum
from typing import Literal, Optional

from pydantic import Field, field_validator

from inference_orchestrator.core.logging import get_logger

from ..base_model_properties import BaseModelProperties, TritonModelProperties

logger = get_logger(__name__)


class PoolingMethod(str, Enum):
    Mean = "mean"
    CLS = "cls"


class HFTritonModelProperties(TritonModelProperties):
    @field_validator("input", mode="after")
    @classmethod
    def _validate_model_input(cls, v: list):
        if len(v) != 3:
            raise ValueError(
                f"Hugging Face models must have exactly 3 inputs. Received {len(v)} inputs."
            )
        names = [inp.name for inp in v]
        expected_names = ["input_ids", "attention_mask", "token_type_ids"]
        if names != expected_names:
            raise ValueError(
                f"Hugging Face models must have inputs named '{expected_names}'. Received {names}."
            )
        return v

    @field_validator("output", mode="after")
    @classmethod
    def _validate_model_output(cls, v: list):
        if len(v) != 1:
            raise ValueError(
                f"Hugging Face models must have exactly 1 output. Received {len(v)} inputs."
            )
        names = [inp.name for inp in v]
        expected_names = ["last_hidden_state"]
        if names != expected_names:
            raise ValueError(
                f"Hugging Face models must have output named {expected_names}. Received {names}."
            )
        return v


class HuggingFaceModelProperties(BaseModelProperties):
    """
    A class to represent the properties of a Hugging Face model.

    Attributes:
        tokens: The token length of the model. It is default to 128.
        type: The type of the model. It should be "hf".
        note: A note about the model. It is optional.
        pooling_method: The pooling method for the model. It should be one of the values in the PoolingMethod enum.
    """

    tokens: int = 128
    note: Optional[str] = None
    pooling_method: PoolingMethod = Field(..., alias="poolingMethod")
    type: Literal["hf"]
    triton_text_encoder_properties: HFTritonModelProperties = Field(
        ..., alias="tritonTextEncoderProperties"
    )
