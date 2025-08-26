import json
from typing import Any, Union, List, Dict, Optional, NewType

from pydantic.v1 import BaseModel, validator, ValidationError, root_validator, Field, conlist

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


class SearchContextDocumentsParameters(BaseModel):
    tensor_fields: Optional[List[str]] = Field(None, alias='tensorFields')
    exclude_input_documents: bool = Field(True, alias='excludeInputDocuments')
    allow_missing_documents: bool = Field(False, alias='allowMissingDocuments')
    allow_missing_embeddings: bool = Field(False, alias='allowMissingEmbeddings')

    @validator('tensor_fields', pre=True, always=True)
    def check_tensor_fields_not_empty(cls, v):
        if v == []:
            raise ValueError('context document tensorFields parameter must be non-empty list.'
                                  ' If you want to use all tensor fields, do not define this parameter.')
        return v


class SearchContextDocuments(BaseModel):
    ids: Optional[Dict[str, float]]
    # If not provided, default parameters are created
    parameters: Optional[SearchContextDocumentsParameters] = SearchContextDocumentsParameters()

    @validator('ids', pre=True, always=True)
    def check_ids_not_empty(cls, v):
        if not v:
            raise ValueError('context["documents"]["ids"] must be present and a non-empty dict of '
                                  'document id to weight pairs.')
        return v


class SearchContext(BaseModel):
    tensor: Optional[List[SearchContextTensor]] = Field(
        min_items=1, max_items=64
    )
    documents: Optional[SearchContextDocuments]

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise InvalidArgError(message=e.json())

    # Root validator to confirm either tensor or documents MUST exist
    @root_validator(pre=False, skip_on_failure=True)
    def validate_at_least_one_context_exists(cls, values):
        tensor = values.get('tensor')
        documents = values.get('documents')

        if tensor is None and documents is None:
            raise ValueError('At least 1 form of context (tensor or documents) must be provided')

        return values


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
    