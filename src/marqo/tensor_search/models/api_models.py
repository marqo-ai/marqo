"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from pydantic import BaseModel
from typing import Union, List, Dict, Optional, Any

from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo import errors


class BaseMarqoModel(BaseModel):
     class Config:
         extra: str = "forbid"
     pass


class SearchQuery(BaseMarqoModel):
    q: Optional[Union[str, Dict[str, float]]] = None
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: Union[None, str] = "TENSOR"
    limit: int = 10
    offset: int = 0
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: Union[None, List[str]] = None
    boost: Optional[Dict] = None
    image_download_headers: Optional[Dict] = None
    context: Optional[SearchContext] = None
    scoreModifiers: Optional[ScoreModifier] = None
    modelAuth: Optional[ModelAuth] = None

    @pydantic.validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )
    
    @pydantic.root_validator
    def validate_query_and_context(cls, values):
        """
        Validates that if LEXICAL search, query must be present.
        If TENSOR search, either query or context must be present.
        """
        search_method = values.get("searchMethod")
        query = values.get("q")
        context = values.get("context")

        if search_method == SearchMethod.LEXICAL:
            if query is None:
                raise errors.InvalidArgError("Query must be provided when using lexical search.")
        elif search_method == SearchMethod.TENSOR:
            if query is None and context is None:
                raise errors.InvalidArgError("At least one of query (`q`) or context vectors (`context`) must be provided when using tensor search.")
        return values

    def get_context_tensor(self) -> Optional[List[SearchContextTensor]]:
        """Extract the tensor from the context, if provided"""
        return self.context.tensor if self.context is not None else None
    

class BulkSearchQueryEntity(SearchQuery):
    index: str

    context: Optional[SearchContext] = None
    scoreModifiers: Optional[ScoreModifier] = None
    def to_search_query(self):
        return SearchQuery(**self.dict())


class BulkSearchQuery(BaseMarqoModel):
    queries: List[BulkSearchQueryEntity]


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
