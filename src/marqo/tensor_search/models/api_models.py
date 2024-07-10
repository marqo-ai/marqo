"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""

from typing import Union, List, Dict, Optional

import pydantic
from pydantic import BaseModel, root_validator, validator

from marqo.base_model import ImmutableStrictBaseModel
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.marqo_index import MarqoIndex
from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor


class BaseMarqoModel(BaseModel):
    class Config:
        extra: str = "forbid"

    pass


class CustomVectorQuery(ImmutableStrictBaseModel):
    class CustomVector(ImmutableStrictBaseModel):
        content: Optional[str] = None
        vector: List[float]

    customVector: CustomVector


class SearchQuery(BaseMarqoModel):
    q: Optional[Union[str, Dict[str, float], CustomVectorQuery]] = None
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: SearchMethod = SearchMethod.TENSOR
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
    scoreModifiers: Optional[ScoreModifierLists] = None
    modelAuth: Optional[ModelAuth] = None
    textQueryPrefix: Optional[str] = None
    hybridParameters: Optional[HybridParameters] = None

    @validator("searchMethod", pre=True)
    def _preprocess_search_method(cls, value):
        """Preprocess the searchMethod value for validation.

        1. Set the default search method to SearchMethod.TENSOR if None is provided.
        2. Return the search method in uppercase if it is a string.
        """
        if value is None:
            return SearchMethod.TENSOR
        elif isinstance(value, str):
            return value.upper()
        else:
            return value

    @root_validator(pre=False, skip_on_failure=True)
    def validate_query_and_context(cls, values):
        """Validate that one of query and context are present for tensor/hybrid search, or just the query for lexical search.

        Raises:
            InvalidArgError: If validation fails
        """
        search_method = values.get('searchMethod')
        query = values.get('q')
        context = values.get('context')

        if search_method.upper() in {SearchMethod.TENSOR, SearchMethod.HYBRID}:
            if query is None and context is None:
                raise ValueError(f"One of Query(q) or context is required for {search_method} "
                                 f"search but both are missing")
        elif search_method.upper() == SearchMethod.LEXICAL:
            if query is None:
                raise ValueError("Query(q) is required for lexical search")
        else:
            raise ValueError(f"Invalid search method {search_method}")
        return values

    @root_validator(pre=False)
    def validate_hybrid_parameters_only_for_hybrid_search(cls, values):
        """Validate that hybrid parameters are only provided for hybrid search"""
        hybrid_parameters = values.get('hybridParameters')
        search_method = values.get('searchMethod')
        if hybrid_parameters is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"Hybrid parameters can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
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
    scoreModifiers: Optional[ScoreModifierLists] = None
    text_query_prefix: Optional[str] = None

    def to_search_query(self):
        return SearchQuery(**self.dict())


class BulkSearchQuery(BaseMarqoModel):
    queries: List[BulkSearchQueryEntity]


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
