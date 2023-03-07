import json
from pydantic import BaseModel
from typing import Any, Union, List, Dict, Optional, NewType, Literal

Qidx = NewType('Qidx', int) # Indicates the position of a search query in a bulk search request
JHash = NewType('JHash', int) # hash of a VectoriseJob. Used for quick access of VectorisedJobs

class VectorisedJobPointer(BaseModel):
    """A VectorisedJobPointer is pointer to a subset of content within a VectorisedJobs (generally from a single query/
    request). `start_idx:end_idx` is a slice to content (or vectors) within a VectorisedJob."""
    job_hash: JHash
    start_idx: int
    end_idx: int

class VectorisedJobs(BaseModel):
    """A vectorised job describes content (e.q. search queries, images) that can be vectorised (i.e can be sent to 
    `s2_inference.vectorise`) in a single batch given they share common inference parameters.

    TODO:
        - pull out image and text content separately.
        -
    """
    model_name: str
    model_properties: Dict[str, Any]
    content: List[Union[str, List[str]]]
    device: str
    normalize_embeddings: bool
    image_download_headers: Optional[Dict]
    content_type: Literal['text', 'image']

    def __hash__(self):
        return self.groupby_key() + hash(json.dumps(self.content, sort_keys=True))

    def groupby_key(self) -> JHash:
        return VectorisedJobs.get_groupby_key(self.model_name, self.model_properties, self.device,
                                              self.normalize_embeddings, self.content_type,
                                              self.image_download_headers)

    @staticmethod
    def get_groupby_key(model_name: str, model_properties: Dict[str, Any], device: str,
                        normalize_embeddings: bool, content_type: str, image_download_headers: Optional[Dict]) -> JHash:
        return JHash(hash(model_name) + hash(json.dumps(model_properties, sort_keys=True))
                     + hash(device) + hash(normalize_embeddings)
                     + hash(content_type)
                     + hash(json.dumps(image_download_headers, sort_keys=True))
                     )

    def add_content(self, content: List[Union[str, List[str]]]) -> VectorisedJobPointer:
        start_idx = len(self.content)
        self.content.extend(content)

        return VectorisedJobPointer(
            job_hash=self.groupby_key(),
            start_idx=start_idx,
            end_idx=len(self.content)
        )
