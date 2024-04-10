"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from pydantic import BaseModel, root_validator
from typing import Union, List, Dict, Optional, Any

from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo.api.exceptions import InvalidArgError
from marqo.core.models.marqo_index import MarqoIndex


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
    efSearch: Optional[int] = None
    approximate: Optional[bool] = None
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: Union[None, List[str]] = None
    boost: Optional[Dict] = None
    image_download_headers: Optional[Dict] = None
    context: Optional[SearchContext] = None
    scoreModifiers: Optional[ScoreModifier] = None
    modelAuth: Optional[ModelAuth] = None

    @root_validator(pre=False)
    def validate_query_and_context(cls, values):
        """Validate that one of query and context are present for tensor search, or just the query for lexical search.

        Raises:
            InvalidArgError: If validation fails
        """
        search_method = values.get('searchMethod')
        query = values.get('q')
        context = values.get('context')

        if search_method.upper() == SearchMethod.TENSOR:
            if query is None and context is None:
                raise ValueError("One of Query(q) or context is required for tensor search but both are missing")
        elif search_method.upper() == SearchMethod.LEXICAL:
            if query is None:
                raise ValueError("Query(q) is required for lexical search")
        else:
            raise ValueError(f"Invalid search method {search_method}")
        return values

    @pydantic.validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )
    
    def get_context_tensor(self) -> Optional[List[SearchContextTensor]]:
        """Extract the tensor from the context, if provided"""
        return self.context.tensor if self.context is not None else None

class BulkSearchQueryEntity(SearchQuery):
    index: MarqoIndex

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
