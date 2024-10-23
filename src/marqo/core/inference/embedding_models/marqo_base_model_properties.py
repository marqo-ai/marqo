from abc import ABC

from pydantic import Field

from marqo.base_model import MarqoBaseModel


class MarqoBaseModelProperties(MarqoBaseModel, ABC):
    """
    The base class for all model properties classes in Marqo.

    Attributes:
        dimensions: The dimensions of the model.
        type: The type of the model
    """
    dimensions: int = Field(..., ge=1)
    type: str
