from abc import ABC

from pydantic import Field

from marqo.base_model import ImmutableBaseModel


class MarqoBaseModelProperties(ImmutableBaseModel, ABC):
    """
    The base class for all model properties classes in Marqo.

    Attributes:
        dimensions: The dimensions of the model.
        type: The type of the model
    """
    dimensions: int = Field(..., ge=1)
    type: str
