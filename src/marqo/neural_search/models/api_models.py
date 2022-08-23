"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from pydantic import BaseModel
from typing import Union, List, Dict
from marqo.neural_search.enums import SearchMethod, Device
from marqo.neural_search import validation
from marqo.neural_search import configs


class SearchQuery(BaseModel):
    q: str
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: Union[None, str] = SearchMethod.NEURAL
    limit: int = 10
    showHighlights: bool = True

    @pydantic.validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )

    class Config:
        arbitrary_types_allowed = True


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
