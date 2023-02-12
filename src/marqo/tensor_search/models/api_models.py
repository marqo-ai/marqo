"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from pydantic import BaseModel
from typing import Union, List, Dict, Optional
from marqo.tensor_search.enums import SearchMethod, Device
from marqo.tensor_search import validation
from marqo.tensor_search import configs


class SearchQuery(BaseModel):
    q: Union[str, dict]
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: Union[None, str] = "TENSOR"
    limit: int = 10
    offset: int = 0
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: List[str] = None
    boost: Optional[Dict] = None

    @pydantic.validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
