import json
from typing import Any, Union, List, Dict, Optional, NewType

from pydantic import BaseModel, validator, ValidationError

from marqo.api.exceptions import InvalidArgError
from marqo.core.inference.api import Modality

from marqo.tensor_search.models.private_models import ModelAuth

Qidx = NewType('Qidx', int) # Indicates the position of a search query in a bulk search request
JHash = NewType('JHash', int) # hash of a VectoriseJob. Used for quick access of VectorisedJobs

class VectorisedJobPointer(BaseModel):
    """A VectorisedJobPointer is pointer to a subset of content within a VectorisedJobs (generally from a single query/
    request). `start_idx:end_idx` is a slice to content (or vectors) within a VectorisedJob."""
    job_hash: JHash
    start_idx: int
    end_idx: int

class VectorisedJobs(BaseModel):
    """A vectorised job describes content (e.q. search queries, images, video, audio) that can be vectorised (i.e can be sent to 
    `s2_inference.vectorise`) in a single batch given they share common inference parameters.

    """
    model_name: str
    model_properties: Dict[str, Any]
    content: List[Union[str, List[str]]]
    device: Optional[str]
    normalize_embeddings: bool
    media_download_headers: Optional[Dict]
    model_auth: Optional[ModelAuth]
    modality: Modality

    def __hash__(self):
        return self.groupby_key() + hash(json.dumps(self.content, sort_keys=True))

    def groupby_key(self) -> JHash:
        return VectorisedJobs.get_groupby_key(self.model_name, self.model_properties, self.device,
                                              self.normalize_embeddings, self.modality,
                                              self.media_download_headers)

    @staticmethod
    def get_groupby_key(model_name: str, model_properties: Dict[str, Any], device: str,
                        normalize_embeddings: bool, modality: str, media_download_headers: Optional[Dict]) -> JHash:
        return JHash(hash(model_name) + hash(json.dumps(model_properties, sort_keys=True))
                     + hash(device) + hash(normalize_embeddings)
                     + hash(modality)
                     + hash(json.dumps(media_download_headers, sort_keys=True))
                     )

    def add_content(self, content: List[Union[str, List[str]]]) -> VectorisedJobPointer:
        start_idx = len(self.content)
        self.content.extend(content)

        return VectorisedJobPointer(
            job_hash=self.groupby_key(),
            start_idx=start_idx,
            end_idx=len(self.content)
        )

class SearchContextTensor(BaseModel):
    vector: List[float]
    weight: float


class SearchContext(BaseModel):
    tensor: List[SearchContextTensor]

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise InvalidArgError(message=e.json())

    @validator('tensor', pre=True, always=True)
    def check_vector_length(cls, v):
        if not (1 <= len(v) <= 64):
            raise InvalidArgError('The number of tensors must be between 1 and 64')
        return v


class QueryContent(BaseModel):
    content: str
    modality: Modality


class QueryContentCollector(BaseModel):
    queries: List[QueryContent]
    @property
    def text_queries(self) -> List[QueryContent]:
        return [q for q in self.queries if q.modality == Modality.TEXT]
    
    @property
    def image_queries(self) -> List[QueryContent]:
        return [q for q in self.queries if q.modality == Modality.IMAGE]
    
    @property
    def video_queries(self) -> List[QueryContent]:
        return [q for q in self.queries if q.modality == Modality.VIDEO]
    
    @property
    def audio_queries(self) -> List[QueryContent]:
        return [q for q in self.queries if q.modality == Modality.AUDIO]
    