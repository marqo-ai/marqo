"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import json
import pydantic
from pydantic import BaseModel
from typing import Any, Union, List, Dict, Optional
from marqo.tensor_search.enums import SearchMethod, Device
from marqo.tensor_search import validation


class SearchQuery(BaseModel):
    q: Union[str, Dict[str, float]]
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: Union[None, str] = "TENSOR"
    limit: int = 10
    offset: int = 0
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: List[str] = None
    boost: Optional[Dict] = None
    image_download_headers: Optional[Dict] = None

    @pydantic.validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )


class VectorisedJobs(BaseModel):
    model_name: str
    model_properties: Dict[str, Any]
    content: List[Union[str, List[str]]]
    device: str
    normalize_embeddings: bool
    image_download_headers: Optional[Dict]

    def __hash__(self):
        return hash(self.model_name) + hash(json.dumps(self.model_properties, sort_keys=True)) + hash(json.dumps(self.content, sort_keys=True)) + hash(self.device) + hash(self.normalize_embeddings) + hash(json.dumps(self.image_download_headers, sort_keys=True))

class BulkSearchQueryEntity(SearchQuery):
    index: str

    def to_search_query(self):
        return SearchQuery(**self.dict())


class BulkSearchQuery(BaseModel):
    queries: List[BulkSearchQueryEntity]


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
