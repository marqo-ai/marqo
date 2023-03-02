"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import json
import pydantic
from pydantic import BaseModel
from typing import Any, Union, List, Dict, Optional, NewType
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


Qidx = NewType('Qidx', int)
JHash = NewType('JHash', int)

class VectorisedJobPointer(BaseModel):
    job_hash: JHash
    start_idx: int
    end_idx: int

class VectorisedJobs(BaseModel):
    model_name: str
    model_properties: Dict[str, Any]
    content: List[Union[str, List[str]]]
    device: str
    normalize_embeddings: bool
    image_download_headers: Optional[Dict]

    def __hash__(self):
        return self.groupby_key() + hash(json.dumps(self.content, sort_keys=True))

    def groupby_key(self) -> JHash:
        return VectorisedJobs.get_groupby_key(self.model_name, self.model_properties, self.device, self.normalize_embeddings, self.image_download_headers)

    @staticmethod
    def get_groupby_key(model_name: str, model_properties: Dict[str, Any], device: str,
                        normalize_embeddings: bool, image_download_headers: Optional[Dict]) -> JHash:
        return JHash(hash(model_name) + hash(json.dumps(model_properties, sort_keys=True))
                     + hash(device) + hash(normalize_embeddings)
                     + hash(json.dumps(image_download_headers, sort_keys=True)))

    def add_content(self, content: List[Union[str, List[str]]]) -> VectorisedJobPointer:
        start_idx = len(self.content)
        self.content.extend(content)

        return VectorisedJobPointer(
            job_hash=self.groupby_key(),
            start_idx=start_idx,
            end_idx=len(self.content)
        )

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
